#include <iostream>
#include <stdexcept>
#include <math.h> 
#include <cmath>
#include <mpi.h>
#include <string>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <stdlib.h>
#include <cuda_runtime.h>

#include <sstream>
#include <string>

using namespace std;

#define SAFE_CUDA(err)\
    if (err != cudaSuccess){ \
        throw std::runtime_error(cudaGetErrorString(err)); \
    }

#define CUDA_CHECK_ERROR\
    if (cudaPeekAtLastError() != cudaSuccess){ \
        throw std::runtime_error(cudaGetErrorString(cudaGetLastError())); \
    }

__device__ double gpu_F(const double x, const double y) {
    return (x*x + y*y) / ((1.0 + 1.0*x*y)*(1.0 + 1.0*x*y));
}

__device__ double gpu_phi(const double x, const double y) {
    return logf(1.0 + 1.0*x*y);
}

double f_grid(const double t) {
	double q = 1.5;
	if (t < 0 || t > 1)
		throw std::runtime_error("Error in computing 'f_grid' function");
	return (pow(1.0 + t, q) - 1.0) / (pow(2.0, q) - 1.0);
}

void compute_grid_processes_number(const int& size, int& x_proc_num, int& y_proc_num) {
    if (size >= 512) {
        x_proc_num = 16;
        y_proc_num = 32;
    } else if (size >= 256) {
        x_proc_num = 16;
        y_proc_num = 16;
    } else if (size >= 128) {
        x_proc_num = 8;
        y_proc_num = 16;
    } else if (size >= 64) {
        x_proc_num = 8;
        y_proc_num = 8;
    } else if (size >= 32) {
        x_proc_num = 4;
        y_proc_num = 8;
    } else if (size >= 16) {
        x_proc_num = 4;
        y_proc_num = 4;
    } else if (size >= 8){
        x_proc_num = 2;
        y_proc_num = 4;
    } else if (size >= 4) {
        x_proc_num = 2;
        y_proc_num = 2;
    } else if (size >= 2) {
        x_proc_num = 1;
        y_proc_num = 2;
    } else if (size >= 1) {
        x_proc_num = 1;
        y_proc_num = 1;
    } else {
        throw std::runtime_error("Incorrect processes number");
    }
}

struct GridParameters {
	int rank, N1, N2, p1, p2, x_index_from, x_index_to, y_index_from, y_index_to;
	int threadsPerBlock, blocksPerGrid, numElements;
	double *x_grid, *y_grid, *product_buf, *norm_buf;
	double eps;
    bool top, bottom, left, right;
    double *hxhy, *gp_x_grid, *gp_y_grid, *gp_x_h_step, *gp_y_h_step;
    bool *gp_is_local_border, *gp_is_global_border;

    double *send_message_top, *send_message_bottom, *send_message_left, *send_message_right;
    double *recv_message_top, *recv_message_bottom, *recv_message_left, *recv_message_right;
    MPI_Request* send_requests;
    MPI_Request* recv_requests;
    cudaStream_t* cuda_streams;
    MPI_Comm comm;

	GridParameters (int rank, MPI_Comm comm, double* x_grid, double* y_grid, int N1, int N2, int p1, int p2, double eps):
		rank (rank), comm (comm), x_grid (x_grid), y_grid (y_grid), 
		send_message_top (NULL), send_message_bottom (NULL), send_message_left (NULL), send_message_right (NULL),
		recv_message_top (NULL), recv_message_bottom (NULL), recv_message_left (NULL), recv_message_right (NULL),
		send_requests (NULL), recv_requests (NULL),
		N1 (N1), N2 (N2),p1 (p1), p2 (p2), eps (eps), 
		x_index_from (0), x_index_to (0), y_index_from (0), y_index_to (0),
		top (false), bottom (false), left (false), right (false) {
            send_requests = new MPI_Request [4];
            recv_requests = new MPI_Request [4];
            //cuda_streams = new cudaStream_t [4];

			int step1, step2;
			step1 = int(floor(1.0 * N1 / p1));
			step2 = int(floor(1.0 * N2 / p2));
			x_index_from = int(floor(1.0 * step1 * floor(1.0 * rank / p2)));
			y_index_from = int(floor((double(rank % p2)) * step2));

			if ((rank + 1) % p2 == 0)
				y_index_to = N2;
			else
				y_index_to = y_index_from + step2; 

			if (rank >= (p1-1)*p2)
				x_index_to = N1;
			else
				x_index_to = x_index_from + step1;

			if (x_index_from == 0)
				top = true;
			if (y_index_from == 0)
				left = true;
			if (y_index_to == N1)
				right = true;
			if (x_index_to == N1)
				bottom = true;

			int size = get_num_x_points() * get_num_y_points();
			SAFE_CUDA(cudaHostAlloc(&hxhy, size * sizeof(double), cudaHostAllocMapped));
			for (int i=0; i<get_num_x_points(); i++){
	        	for (int j=0; j<get_num_y_points(); j++){
	        		int grid_i, grid_j;
    				get_real_grid_index(i, j, grid_i, grid_j);
    				if (not is_border_point(grid_i, grid_j))
	        			hxhy[i*get_num_y_points()+j] = ((get_x_h_step(grid_i) + get_x_h_step(grid_i-1)) / 2.0) * ((get_y_h_step(grid_j) + get_y_h_step(grid_j-1)) / 2.0);
	        		else
	        			hxhy[i*get_num_y_points()+j] = 0.0;
	        	}
	        }

            SAFE_CUDA(cudaMalloc(&gp_x_grid, size * sizeof(double)));
            SAFE_CUDA(cudaMalloc(&gp_y_grid, size * sizeof(double)));
            SAFE_CUDA(cudaMalloc(&gp_is_global_border, size * sizeof(bool)));
            SAFE_CUDA(cudaMalloc(&gp_is_local_border, size * sizeof(bool)));
            SAFE_CUDA(cudaMalloc(&gp_x_h_step, size * sizeof(double)));
            SAFE_CUDA(cudaMalloc(&gp_y_h_step, size * sizeof(double)));

            double* l_gp_x_grid = new double [size];
            double* l_gp_y_grid = new double [size];
            bool* l_gp_is_global_border = new bool [size];
            bool* l_gp_is_local_border = new bool [size];
            double* l_gp_x_h_step = new double [size];
            double* l_gp_y_h_step = new double [size];

			for (int i=0; i<get_num_x_points(); i++) {
		    	for (int j=0; j<get_num_y_points(); j++) {
		    		int grid_i, grid_j;
		    		get_real_grid_index(i, j, grid_i, grid_j);
		    		l_gp_x_grid[i*get_num_y_points()+j] = get_x_grid_value(grid_i);
		    		l_gp_y_grid[i*get_num_y_points()+j] = get_y_grid_value(grid_j);

		    		if (is_border_point(grid_i, grid_j))
						l_gp_is_global_border[i*get_num_y_points()+j] = true;
					else
						l_gp_is_global_border[i*get_num_y_points()+j] = false;

					if ((i < get_num_x_points() - 1) && (j < get_num_y_points() - 1)) {
						l_gp_x_h_step[i*get_num_y_points()+j] = get_x_h_step(grid_i);
						l_gp_y_h_step[i*get_num_y_points()+j] = get_y_h_step(grid_j);
					}
					else {
                        l_gp_x_h_step[i*get_num_y_points()+j] = 0.0;
                        l_gp_y_h_step[i*get_num_y_points()+j] = 0.0;
					}

					if ((i == 0) || (j == 0) || (i == get_num_x_points() - 1) || (j == get_num_y_points() - 1))
						l_gp_is_local_border[i*get_num_y_points()+j] = true;
					else
						l_gp_is_local_border[i*get_num_y_points()+j] = false;
		    	}
			}

			SAFE_CUDA(cudaMemcpy(gp_x_grid, l_gp_x_grid, size * sizeof(double), cudaMemcpyHostToDevice));
			SAFE_CUDA(cudaMemcpy(gp_y_grid, l_gp_y_grid, size * sizeof(double), cudaMemcpyHostToDevice));
			SAFE_CUDA(cudaMemcpy(gp_is_global_border, l_gp_is_global_border, size * sizeof(bool), cudaMemcpyHostToDevice));
			SAFE_CUDA(cudaMemcpy(gp_is_local_border, l_gp_is_local_border, size * sizeof(bool), cudaMemcpyHostToDevice));
			SAFE_CUDA(cudaMemcpy(gp_x_h_step, l_gp_x_h_step, size * sizeof(double), cudaMemcpyHostToDevice));
			SAFE_CUDA(cudaMemcpy(gp_y_h_step, l_gp_y_h_step, size * sizeof(double), cudaMemcpyHostToDevice));
			
			threadsPerBlock = 128;
			numElements = get_num_x_points() * get_num_y_points();
			blocksPerGrid = 1.0 * (numElements + threadsPerBlock - 1) / threadsPerBlock;
            // need for compute scalar product
            SAFE_CUDA(cudaMalloc(&product_buf, (blocksPerGrid+1)*sizeof(double)));
            // need for compute norm
            SAFE_CUDA(cudaMalloc(&norm_buf, (blocksPerGrid+1)*sizeof(double)));
		}

	int get_num_x_points() {
		if (bottom) 
			return x_index_to - x_index_from + 1;
		else
			return x_index_to - x_index_from;
	}

	int get_num_y_points() {
		if (right) 
			return y_index_to - y_index_from + 1;
		else
			return y_index_to - y_index_from;
	}

	void get_real_grid_index(int i, int j, int& grid_i, int& grid_j) {
		grid_i = x_index_from+i;
		grid_j = y_index_from+j;
	}

	double get_x_grid_value(int grid_i) {
		return x_grid[grid_i];
	}

	double get_y_grid_value(int grid_j) {
		return y_grid[grid_j];
	}

	double get_x_h_step(int grid_i) {
		return x_grid[grid_i+1] - x_grid[grid_i];
	}

	double get_y_h_step(int grid_j) {
		return y_grid[grid_j+1] - y_grid[grid_j];
	}

	int get_top_rank() {
		return rank - p2;
	}

	int get_bottom_rank() {
		return rank + p2;
	}

	int get_left_rank() {
		return rank - 1;
	}

	int get_right_rank() {
		return rank + 1;
	}

	bool is_border_point(int grid_i, int grid_j) {
		if ((grid_i == 0) || (grid_j == 0) || (grid_i == N1) || (grid_j == N2))
			return true;
		else
			return false;
	}
};

__global__ void gpu_scalar_product(double *f1, double *f2, double* hxhy, double *results, int n) {
	extern __shared__ double sdata[];
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	// load input into __shared__ memory
	double x = 0.0;
	if(i < n)
		x = hxhy[i] * f1[i] * f2[i];//input[i];

	sdata[tx] = x;
	__syncthreads(); 
	// block-wide reduction in __shared__ mem
	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
		if(tx < offset) {
			// add a partial sum upstream to our own
			sdata[tx] += sdata[tx + offset];
		}
		__syncthreads();
	}
	// finally, thread 0 writes the result
	if(threadIdx.x == 0) {
		// note that the result is per-block
		// not per-thread
		results[blockIdx.x] = sdata[0];
	}
}


double scalar_product(GridParameters gp, double* f1, double* f2) {	
    // reduce per-block partial sums
    gpu_scalar_product<<<gp.blocksPerGrid, gp.threadsPerBlock, gp.threadsPerBlock * sizeof(double)>>>(f1, f2, gp.hxhy, gp.product_buf, gp.numElements); CUDA_CHECK_ERROR;

	// reduce partial sums to a total sum
    double *tmp_product_buf = new double [gp.blocksPerGrid];
	SAFE_CUDA(cudaMemcpy(tmp_product_buf, gp.product_buf, gp.blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost));
	double gpu_product = 0.0;
	for (int i=0; i<gp.blocksPerGrid; i++)
		gpu_product += tmp_product_buf[i];

    double global_product = 0.0;
    int status = MPI_Allreduce(&gpu_product, &global_product, 1, MPI_DOUBLE, MPI_SUM, gp.comm);
    if (status != MPI_SUCCESS) throw std::runtime_error("Error in compute scalar_product!");
    //printf("rank %d: global_product=%f gpu_product=%f gpu_product_old=%f\n", gp.rank, global_product, gpu_product, gpu_product_old);
    return global_product;
}


enum MPI_tags { SendToTop, SendToBottom, SendToLeft, SendToRight};


__global__ void gpu_copy_send_message(double* send_buffer, const double* func, int num_y_points, int numElements){
    int no_thread = threadIdx.x + blockIdx.x * blockDim.x;

    if (no_thread < numElements) {
        send_buffer[no_thread] = func[no_thread * num_y_points];
    }
}


__device__ double gpu_compute_delta(double f_curr, double f_top, double f_bottom, double f_left,
                                  double f_right, double* gp_x_h_step, double* gp_y_h_step, int no_thread, int num_y_points) {
    double h_i_1 = gp_x_h_step[no_thread-num_y_points];
    double h_i = gp_x_h_step[no_thread];
    double h_j_1 = gp_y_h_step[no_thread-1];
    double h_j = gp_y_h_step[no_thread];
    double average_hx = (h_i + h_i_1) / 2.0;
    double average_hy = (h_j + h_j_1) / 2.0;

    return (1.0 / average_hx) * ((f_curr - f_top) / h_i_1 - (f_bottom - f_curr) / h_i) +
            (1.0 / average_hy) * ((f_curr - f_left) / h_j_1 - (f_right - f_curr) / h_j);
}


__global__ void gpu_compute_approx_delta(double *delta_func, double *func, double *x_h_step, double *y_h_step,
	    bool *is_local_border, bool *is_global_border,  int num_x_points, int num_y_points, int numElements,
	    bool top, bool bottom, bool left, bool right,
	    double *recv_message_top, double *recv_message_bottom, double *recv_message_left, double *recv_message_right)
{
	int no_thread = threadIdx.x + blockDim.x * blockIdx.x;
	double f_top, f_bottom, f_left, f_right, f_curr;

	if ((no_thread < numElements) && (is_global_border[no_thread] == false)) {
	    if (is_local_border[no_thread] == true) {
            f_top = func[no_thread-num_y_points];
            f_bottom = func[no_thread+num_y_points];
            f_left = func[no_thread-1];
            f_right = func[no_thread+1];
            f_curr = func[no_thread];

	        // border points (except corner)
            if ((not top) && (0 < no_thread) && (no_thread < num_y_points - 1))
                delta_func[no_thread] = gpu_compute_delta(f_curr, recv_message_top[no_thread], f_bottom, f_left, f_right, x_h_step, x_h_step, no_thread, num_y_points);
            if ((not bottom) && (no_thread > (num_x_points-1)*num_y_points + 1) && (no_thread < num_x_points*num_y_points - 1)) {
                int index = no_thread - (num_x_points-1)*num_y_points;
                delta_func[no_thread] = gpu_compute_delta(f_curr, f_top, recv_message_bottom[index], f_left, f_right, x_h_step, x_h_step, no_thread, num_y_points);
            }
            if ((not left) && (no_thread % num_y_points == 0) && (no_thread != 0) && (no_thread != ((num_x_points - 1)*num_y_points))) {
                int index = no_thread % num_y_points;
                delta_func[no_thread] = gpu_compute_delta(f_curr, f_top, f_bottom, recv_message_left[index], f_right, x_h_step, x_h_step, no_thread, num_y_points);
            }
            if ((not right) && ((no_thread + 1) % num_y_points == 0) && (no_thread != num_y_points - 1) && (no_thread != num_x_points*num_y_points - 1)) {
                int index = (no_thread + 1) % num_y_points;
                delta_func[no_thread] = gpu_compute_delta(f_curr, f_top, f_bottom, f_left, recv_message_right[index], x_h_step, x_h_step, no_thread, num_y_points);
            }

            // corner points
            if ((not top) && (not left) && ((no_thread == 0))) {
                delta_func[no_thread] = gpu_compute_delta(f_curr, recv_message_top[no_thread], f_bottom, recv_message_left[no_thread], f_right, x_h_step, x_h_step, no_thread, num_y_points);
            }
            if ((not top) && (not right) && ((no_thread == num_y_points - 1))) {
                int index1 = no_thread;
                int index2 = (no_thread + 1) % num_y_points;;
                delta_func[no_thread] = gpu_compute_delta(f_curr, recv_message_top[index1], f_bottom, f_left, recv_message_right[index2], x_h_step, x_h_step, no_thread, num_y_points);
            }
            if ((not bottom) && (not left) && (no_thread == (num_x_points - 1)*num_y_points)) {
                int index1 = no_thread - (num_x_points-1)*num_y_points;
                int index2 = no_thread % num_y_points;
                delta_func[no_thread] = gpu_compute_delta(f_curr, f_top, recv_message_bottom[index1], recv_message_left[index2], f_right, x_h_step, x_h_step, no_thread, num_y_points);
            }
            if ((not bottom) && (not right) && (no_thread == (num_x_points*num_y_points - 1))) {
                int index = no_thread - (num_x_points-1)*num_y_points;;
                delta_func[no_thread] = gpu_compute_delta(f_curr, f_top, recv_message_bottom[index], f_left, recv_message_right[index], x_h_step, x_h_step, no_thread, num_y_points);
            }
	    }
	    else {
	        // inner points
	        delta_func[no_thread] = gpu_compute_delta(func[no_thread], func[no_thread-num_y_points], func[no_thread+num_y_points],
	               func[no_thread-1], func[no_thread+1], x_h_step, y_h_step, no_thread, num_y_points);
	    }
    }
}


void compute_approx_delta(GridParameters gp, double* delta_func, double* func) {

    if (gp.send_message_top == NULL)
        SAFE_CUDA(cudaHostAlloc(&gp.send_message_top, gp.get_num_y_points() * sizeof(double), cudaHostAllocMapped));
    if (gp.send_message_bottom == NULL)
        SAFE_CUDA(cudaHostAlloc(&gp.send_message_bottom, gp.get_num_y_points() * sizeof(double), cudaHostAllocMapped));
    if (gp.send_message_left == NULL)
        SAFE_CUDA(cudaHostAlloc(&gp.send_message_left, gp.get_num_x_points() * sizeof(double), cudaHostAllocMapped));
    if (gp.send_message_right == NULL)
        SAFE_CUDA(cudaHostAlloc(&gp.send_message_right, gp.get_num_x_points() * sizeof(double), cudaHostAllocMapped));

    if (gp.recv_message_top == NULL)
        SAFE_CUDA(cudaHostAlloc(&gp.recv_message_top, gp.get_num_y_points() * sizeof(double), cudaHostAllocMapped));
    if (gp.recv_message_bottom == NULL)
        SAFE_CUDA(cudaHostAlloc(&gp.recv_message_bottom, gp.get_num_y_points() * sizeof(double), cudaHostAllocMapped));
    if (gp.recv_message_left == NULL)
        SAFE_CUDA(cudaHostAlloc(&gp.recv_message_left, gp.get_num_x_points() * sizeof(double), cudaHostAllocMapped));
    if (gp.recv_message_right == NULL)
        SAFE_CUDA(cudaHostAlloc(&gp.recv_message_right, gp.get_num_x_points() * sizeof(double), cudaHostAllocMapped));
	
	// copy data to send from gpu to host
	SAFE_CUDA(cudaMemcpy(gp.send_message_top, func, gp.get_num_y_points() * sizeof(double), cudaMemcpyHostToDevice));
	SAFE_CUDA(cudaMemcpy(gp.send_message_bottom, func + (gp.get_num_x_points()-1)*gp.get_num_y_points(), 
		gp.get_num_y_points() * sizeof(double), cudaMemcpyHostToDevice));
	gpu_copy_send_message<<<gp.get_num_x_points(), 1, 0>>>(gp.send_message_left, func, gp.get_num_y_points(), gp.numElements); CUDA_CHECK_ERROR;
	gpu_copy_send_message<<<gp.get_num_x_points(), 1, 0>>>(gp.send_message_right, func + gp.get_num_y_points() - 1, gp.get_num_y_points(), gp.numElements); CUDA_CHECK_ERROR;
	
    //SAFE_CUDA(cudaMemcpyAsync(gp.send_message_top, func, gp.get_num_y_points() * sizeof(double),
    //        cudaMemcpyDeviceToHost, gp.cuda_streams[0]));
    //SAFE_CUDA(cudaMemcpyAsync(gp.send_message_bottom, func + (gp.get_num_x_points()-1)*gp.get_num_y_points(),
    //        gp.get_num_y_points() * sizeof(double), cudaMemcpyDeviceToHost, gp.cuda_streams[1]));
    //gpu_copy_send_message<<<gp.get_num_x_points(), 1, 0, gp.cuda_streams[2]>>>(gp.send_message_left, func, gp.get_num_y_points(), gp.numElements); CUDA_CHECK_ERROR;
    //gpu_copy_send_message<<<gp.get_num_x_points(), 1, 0, gp.cuda_streams[3]>>>(gp.send_message_right, func + gp.get_num_y_points() - 1, gp.get_num_y_points(), gp.numElements); CUDA_CHECK_ERROR;
	
    int status;
    int send_count=0;
    if (not gp.top) {
        //cudaStreamSynchronize(gp.cuda_streams[0]); CUDA_CHECK_ERROR;
        status = MPI_Isend(gp.send_message_top, gp.get_num_y_points(), MPI_DOUBLE,
            gp.get_top_rank(), SendToTop, gp.comm, &(gp.send_requests[send_count]));
        if (status != MPI_SUCCESS) throw std::runtime_error("Error in send message!");
        send_count++;
    }
    if (not gp.bottom) {
        //cudaStreamSynchronize(gp.cuda_streams[1]); CUDA_CHECK_ERROR;
        status = MPI_Isend(gp.send_message_bottom, gp.get_num_y_points(), MPI_DOUBLE,
            gp.get_bottom_rank(), SendToBottom, gp.comm, &(gp.send_requests[send_count]));
        if (status != MPI_SUCCESS) throw std::runtime_error("Error in send message!");
        send_count++;
    }
    if (not gp.left) {
        //cudaStreamSynchronize(gp.cuda_streams[2]); CUDA_CHECK_ERROR;
        status = MPI_Isend(gp.send_message_left, gp.get_num_x_points(), MPI_DOUBLE,
            gp.get_left_rank(), SendToLeft, gp.comm, &(gp.send_requests[send_count]));
        if (status != MPI_SUCCESS) throw std::runtime_error("Error in send message!");
        send_count++;
    }
    if (not gp.right) {
        //cudaStreamSynchronize(gp.cuda_streams[3]); CUDA_CHECK_ERROR;
        status = MPI_Isend(gp.send_message_right, gp.get_num_x_points(), MPI_DOUBLE,
            gp.get_right_rank(), SendToRight, gp.comm, &(gp.send_requests[send_count]));
        if (status != MPI_SUCCESS) throw std::runtime_error("Error in send message!");
        send_count++;
    }

    int recv_count=0;
    if (not gp.top) {
        status = MPI_Irecv(gp.recv_message_top, gp.get_num_y_points(), MPI_DOUBLE,
            gp.get_top_rank(), SendToBottom, gp.comm, &(gp.recv_requests[recv_count]));
        if (status != MPI_SUCCESS) throw std::runtime_error("Error in receive message!");
        recv_count++;
    }
    if (not gp.bottom) {
        status = MPI_Irecv(gp.recv_message_bottom, gp.get_num_y_points(), MPI_DOUBLE,
            gp.get_bottom_rank(), SendToTop, gp.comm, &(gp.recv_requests[recv_count]));
        if (status != MPI_SUCCESS) throw std::runtime_error("Error in receive message!");
        recv_count++;
    }
    if (not gp.left) {
        status = MPI_Irecv(gp.recv_message_left, gp.get_num_x_points(), MPI_DOUBLE,
            gp.get_left_rank(), SendToRight, gp.comm, &(gp.recv_requests[recv_count]));
        if (status != MPI_SUCCESS) throw std::runtime_error("Error in receive message!");
        recv_count++;
    }
    if (not gp.right) {
        status = MPI_Irecv(gp.recv_message_right, gp.get_num_x_points(), MPI_DOUBLE,
            gp.get_right_rank(), SendToLeft, gp.comm, &(gp.recv_requests[recv_count]));
        if (status != MPI_SUCCESS) throw std::runtime_error("Error in receive message!");
        recv_count++;
    }

    status = MPI_Waitall(recv_count, gp.recv_requests, MPI_STATUS_IGNORE);
    if (status != MPI_SUCCESS) throw std::runtime_error("Error in waiting receive message!");

    status = MPI_Waitall(send_count, gp.send_requests, MPI_STATUS_IGNORE);
    if (status != MPI_SUCCESS) throw std::runtime_error("Error in waiting send message!");

    gpu_compute_approx_delta<<<gp.blocksPerGrid, gp.threadsPerBlock>>>(delta_func, func, gp.gp_x_h_step, gp.gp_y_h_step,
        gp.gp_is_local_border, gp.gp_is_global_border, gp.get_num_x_points(), gp.get_num_y_points(), gp.numElements,
        gp.top, gp.bottom, gp.left, gp.right, gp.recv_message_top, gp.recv_message_bottom,
        gp.recv_message_left, gp.recv_message_right); CUDA_CHECK_ERROR;
}


__global__ void gpu_compute_r(double *r, const double *delta_p, double *x_grid, double *y_grid, bool *is_global_border, int numElements) {
  int no_thread = threadIdx.x + blockDim.x * blockIdx.x;

  if (no_thread < numElements) {
    if (is_global_border[no_thread] == true)
        r[no_thread] = 0.0;
    else
  	    r[no_thread] = delta_p[no_thread] - gpu_F(x_grid[no_thread], y_grid[no_thread]);
  }
}

void compute_r(GridParameters gp, double *r, const double *delta_p) {
    gpu_compute_r<<<gp.blocksPerGrid, gp.threadsPerBlock>>>(r, delta_p, gp.gp_x_grid, gp.gp_y_grid, gp.gp_is_global_border, gp.numElements); CUDA_CHECK_ERROR;
}

__global__ void gpu_compute_g(double *g, double *r, double alpha, int numElements) {
  int no_thread = threadIdx.x + blockDim.x * blockIdx.x;

  if (no_thread < numElements) {
  	g[no_thread] = r[no_thread] - alpha * g[no_thread];
  }
}


void compute_g(GridParameters gp, double *g, double *r, double alpha) {
    gpu_compute_g<<<gp.blocksPerGrid, gp.threadsPerBlock>>>(g, r, alpha, gp.numElements); CUDA_CHECK_ERROR;
}


__global__ void gpu_compute_p(double *p, double *p_prev, double *g, double tau, int numElements) {
  int no_thread = threadIdx.x + blockDim.x * blockIdx.x;

  if (no_thread < numElements) {
  	p[no_thread] = p_prev[no_thread] - tau * g[no_thread];
  }
}


void compute_p(GridParameters gp, double *p, double* p_prev, double *g, double tau) {
    gpu_compute_p<<<gp.blocksPerGrid, gp.threadsPerBlock>>>(p, p_prev, g, tau, gp.numElements); CUDA_CHECK_ERROR;
}


__global__ void gpu_norm(double *p, double *p_prev, double *results, int n) {
	extern __shared__ double sdata[];
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	// load input into __shared__ memory
	double x = 0.0;
	if(i < n)
		x = abs(p[i]-p_prev[i]);

	sdata[tx] = x;
	__syncthreads(); 
	// block-wide reduction in __shared__ mem
	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
		if(tx < offset) {
			// add a partial sum upstream to our own
			sdata[tx] = max(sdata[tx], sdata[tx + offset]);
		}
		__syncthreads();
	}
	// finally, thread 0 writes the result
	if(threadIdx.x == 0) {
		// note that the result is per-block
		// not per-thread
		results[blockIdx.x] = sdata[0];
	}
}


double compute_norm(GridParameters gp, double *p, double *p_prev) {	
    // reduce per-block partial sums
    gpu_norm<<<gp.blocksPerGrid, gp.threadsPerBlock, gp.threadsPerBlock * sizeof(double)>>>(p, p_prev, gp.norm_buf, gp.numElements); CUDA_CHECK_ERROR;

     // reduce partial sums to a total sum
	double *tmp_norm_buf = new double [gp.blocksPerGrid];
	SAFE_CUDA(cudaMemcpy(tmp_norm_buf, gp.norm_buf, gp.blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost));
	double gpu_norm = 0.0;
	for (int i=0; i<gp.blocksPerGrid; i++)
		gpu_norm = max(gpu_norm, tmp_norm_buf[i]);

 	double global_norm = 0.0;
 	int status = MPI_Allreduce(&gpu_norm, &global_norm, 1, MPI_DOUBLE, MPI_MAX, gp.comm);
     if (status != MPI_SUCCESS) throw std::runtime_error("Error in compute scalar_product!");
     //printf("rank %d: gpu_norm=%f global_norm=%f real_norm=%f\n", gp.rank, gpu_norm, global_norm, real_norm);
     return global_norm;
}


__global__ void gpu_init_vector(double *func, int numElements) {
    int no_thread = threadIdx.x + blockDim.x * blockIdx.x;

    if (no_thread < numElements) {
        func[no_thread] = 0.0;
    }
}


void init_vector(GridParameters gp, double* func) {
    gpu_init_vector<<<gp.blocksPerGrid, gp.threadsPerBlock>>>(func, gp.numElements); CUDA_CHECK_ERROR;
}


__global__ void gpu_init_p_prev(double *p_prev, double *x_grid, double *y_grid, bool *is_global_border, int numElements) {
    int no_thread = threadIdx.x + blockDim.x * blockIdx.x;

    if (no_thread < numElements) {
        if (is_global_border[no_thread] == false)
            p_prev[no_thread] = 0.0;
        else
            p_prev[no_thread] = gpu_phi(x_grid[no_thread], y_grid[no_thread]);
    }
}


void init_p_prev(GridParameters gp, double* p_prev) {
    gpu_init_p_prev<<<gp.blocksPerGrid, gp.threadsPerBlock>>>(p_prev, gp.gp_x_grid, gp.gp_y_grid, gp.gp_is_global_border, gp.numElements); CUDA_CHECK_ERROR;
}


__global__ void gpu_phi_on_grid(double *phi_on_grid, double *x_grid, double *y_grid, int numElements) {
  int no_thread = threadIdx.x + blockDim.x * blockIdx.x;

  if (no_thread < numElements) {
	  phi_on_grid[no_thread] = gpu_phi(x_grid[no_thread], y_grid[no_thread]);
  }
}


int main (int argc, char** argv) {
	if (argc != 3)
		throw std::runtime_error("Incorrect number of arguments");
	clock_t begin = clock();

	const double A1 = 0.0;
	const double A2 = 3.0;
	const double B1 = 0.0;
	const double B2 = 3.0;

	const int N1 = atoi(argv[1]);
	const int N2 = atoi(argv[2]);
	const double eps = 0.0001;

	double* x_grid = new double [N1+1];
	double* y_grid = new double [N2+1];

	for (int i=0; i<=N1; i++) {
		x_grid[i] = A2 * f_grid(1.0*i/N1) + A1 * (1 - f_grid(1.0*i/N1));
		//std::cout << "x_grid[" << i << "]=" << x_grid[i] << std::endl;
	}
	for (int j=0; j<=N2; j++) {
		y_grid[j] = B2 * f_grid(1.0*j/N2) + B1 * (1 - f_grid(1.0*j/N2));
		//std::cout << "y_grid[" << j << "]=" << y_grid[j] << std::endl;
	}

	int rank, size;
	int p1, p2;

	MPI_Init (&argc, &argv);	/* starts MPI */
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);	/* get current process id */
	MPI_Comm_size (MPI_COMM_WORLD, &size);	/* get number of processes */

	compute_grid_processes_number(size, p1, p2);

	// filter extra processes
	if (rank < p1 * p2) {
		if (rank == 0) {
			std::cout << "p1=" << p1 << " p2=" << p2 << " size=" << size << std::endl;
	    }

	    GridParameters gp(rank, MPI_COMM_WORLD, x_grid, y_grid, N1, N2, p1, p2, eps);
	   	//printf("rank %d: x_index_from = %d  x_index_to = %d  y_index_from = %d y_index_to = %d  top=%d bottom=%d left=%d right=%d\n", 
	    //	gp.rank, gp.x_index_from, gp.x_index_to, gp.y_index_from, gp.y_index_to, gp.top, gp.bottom, gp.left, gp.right);
		if (rank == 0) {
			printf("threadsPerBlock=%d  blocksPerGrid=%d  numElements=%d\n", gp.threadsPerBlock, gp.blocksPerGrid, gp.numElements);
			int device = 0;
			struct cudaDeviceProp properties;
			cudaGetDeviceProperties(&properties, device);
			printf("maxGridSize=%d\n",properties.maxGridSize[0]);
            printf("maxThreadsDim=%d\n",properties.maxThreadsDim[0]);
            printf("maxThreadsPerBlock=%d\n",properties.maxThreadsPerBlock);
		}
		
	    // allocate gpu memory
	    int size = gp.get_num_x_points() * gp.get_num_y_points();
	    double *p, *p_prev, *g, *r, *delta_p, *delta_r, *delta_g;
	    SAFE_CUDA(cudaMalloc(&p, size * sizeof(double)));
	    SAFE_CUDA(cudaMalloc(&p_prev, size * sizeof(double)));
	    SAFE_CUDA(cudaMalloc(&g, size * sizeof(double)));
	    SAFE_CUDA(cudaMalloc(&r, size * sizeof(double)));
	    SAFE_CUDA(cudaMalloc(&delta_p, size * sizeof(double)));
	    SAFE_CUDA(cudaMalloc(&delta_r, size * sizeof(double)));
	    SAFE_CUDA(cudaMalloc(&delta_g, size * sizeof(double)));

	    init_p_prev(gp, p_prev);
	    init_vector(gp, r);
	    init_vector(gp, g);
	    init_vector(gp, delta_p);
	    init_vector(gp, delta_g);
	    init_vector(gp, delta_r);

	    double scalar_product_delta_g_and_g = 1.0;
	    double scalar_product_delta_r_and_g = 1.0;
	    double scalar_product_r_and_g = 1.0;
	    double alpha = 0.0;
	    double tau = 0.0;

        // compute phi (needed for check result)
        double* phi_on_grid;
        SAFE_CUDA(cudaMalloc(&phi_on_grid, size * sizeof(double)));
        gpu_phi_on_grid<<<gp.blocksPerGrid, gp.threadsPerBlock>>>(phi_on_grid, gp.gp_x_grid, gp.gp_y_grid, gp.numElements); CUDA_CHECK_ERROR;

	    int n_iter = 1;
	    while (true) {
	    	compute_approx_delta(gp, delta_p, p_prev); CUDA_CHECK_ERROR;
	    	compute_r(gp, r, delta_p);  CUDA_CHECK_ERROR;

	    	if (n_iter > 1) {
	    		compute_approx_delta(gp, delta_r, r);  CUDA_CHECK_ERROR;
	    		scalar_product_delta_r_and_g = scalar_product(gp, delta_r, g);  CUDA_CHECK_ERROR;
	    		alpha = 1.0 * scalar_product_delta_r_and_g / scalar_product_delta_g_and_g;
	    	}

	    	if (n_iter > 1) { 
	    		compute_g(gp, g, r, alpha);  CUDA_CHECK_ERROR;
			}
	    	else 
            	swap(g, r);

            compute_approx_delta(gp, delta_g, g);  CUDA_CHECK_ERROR;
            if (n_iter > 1) {
            	scalar_product_r_and_g = scalar_product(gp, r, g);  CUDA_CHECK_ERROR;
            }
            else {
            	scalar_product_r_and_g = scalar_product(gp, g, g);  CUDA_CHECK_ERROR;
            }

            scalar_product_delta_g_and_g = scalar_product(gp, delta_g, g);  CUDA_CHECK_ERROR;
	        tau = 1.0 * scalar_product_r_and_g / scalar_product_delta_g_and_g;

	       	compute_p(gp, p, p_prev, g, tau);
	       	double norm_p_p_prev = compute_norm(gp, p, p_prev);  CUDA_CHECK_ERROR;
	       	double norm_p_phi = compute_norm(gp, p, phi_on_grid);  CUDA_CHECK_ERROR;
	       	if (rank == 0)
	       		printf("# iteration %d: norm_p_p_prev=%f norm_p_phi=%f\n", n_iter, norm_p_p_prev, norm_p_phi);
	       	if (norm_p_p_prev < gp.eps)
            	break;

            swap(p, p_prev);
	    	n_iter += 1;
	    }

	    // free gpu memory
        SAFE_CUDA(cudaFree(phi_on_grid));
		SAFE_CUDA(cudaFree(p));
	    SAFE_CUDA(cudaFree(p_prev));
	    SAFE_CUDA(cudaFree(g));
	    SAFE_CUDA(cudaFree(r));
	    SAFE_CUDA(cudaFree(delta_p));
	    SAFE_CUDA(cudaFree(delta_r));
	    SAFE_CUDA(cudaFree(delta_g));
	    
		SAFE_CUDA(cudaFree(gp.gp_x_grid));
		SAFE_CUDA(cudaFree(gp.gp_y_grid));
		SAFE_CUDA(cudaFree(gp.gp_is_global_border));
		SAFE_CUDA(cudaFree(gp.gp_is_local_border));
		SAFE_CUDA(cudaFree(gp.gp_x_h_step));
		SAFE_CUDA(cudaFree(gp.gp_y_h_step));
		SAFE_CUDA(cudaFree(gp.product_buf));
		SAFE_CUDA(cudaFree(gp.norm_buf));
	}
	MPI_Finalize();

	if (rank == 0) {
		clock_t end = clock();
	  	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	  	printf("Algorithm finished! Elapsed time: %f sec\n", elapsed_secs);
	}
	return 0;
}
