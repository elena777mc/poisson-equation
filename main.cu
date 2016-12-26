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

double F(const double x, const double y) {
    double t = 1.0 + 1.0*x*y;
    if (t == 0)
    	throw std::runtime_error("Error in computing 'F' function");
    return (x*x + y*y)/(t*t);
}

double phi(const double x, const double y) {
    double t = 1.0 + 1.0 * x*y;
    if (t <= 0)
        throw std::runtime_error("Error in computing 'phi' function");
    return log(t);
}

__device__ double gpu_F(const double x, const double y) {
    return (x*x + y*y) / ((1.0 + 1.0*x*y)*(1.0 + 1.0*x*y));
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
	double *x_grid, *y_grid;
	double eps;
    bool top, bottom, left, right;
    double *hxhy, *gp_x_grid, *gp_y_grid, *gp_is_not_border, *gp_x_h_step, *gp_y_h_step;

    double *send_message_top, *send_message_bottom, *send_message_left, *send_message_right;
    double *recv_message_top, *recv_message_bottom, *recv_message_left, *recv_message_right;
    MPI_Request* send_requests;
    MPI_Request* recv_requests;
    MPI_Comm comm;

	GridParameters (int rank, MPI_Comm comm, double* x_grid, double* y_grid, int N1, int N2, int p1, int p2, double eps):
		rank (rank), comm (comm), x_grid (x_grid), y_grid (y_grid), 
		send_message_top (NULL), send_message_bottom (NULL), send_message_left (NULL), send_message_right (NULL),
		recv_message_top (NULL), recv_message_bottom (NULL), recv_message_left (NULL), recv_message_right (NULL),
		send_requests (NULL), recv_requests (NULL),
		N1 (N1), N2 (N2),p1 (p1), p2 (p2), eps (eps), 
		x_index_from (0), x_index_to (0), y_index_from (0), y_index_to (0),
		top (false), bottom (false), left (false), right (false) {
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

			hxhy = new double [get_num_x_points() * get_num_y_points()];
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

	        gp_x_grid = new double [get_num_x_points() * get_num_y_points()];
			gp_y_grid = new double [get_num_x_points() * get_num_y_points()];
			gp_is_not_border = new double [get_num_x_points() * get_num_y_points()];
			gp_x_h_step = new double [get_num_x_points() * get_num_y_points()];
			gp_y_h_step = new double [get_num_x_points() * get_num_y_points()];

			for (int i=0; i<get_num_x_points(); i++) {
		    	for (int j=0; j<get_num_y_points(); j++) {
		    		int grid_i, grid_j;
		    		get_real_grid_index(i, j, grid_i, grid_j);
		    		gp_x_grid[i*get_num_y_points()+j] = get_x_grid_value(grid_i);
		    		gp_y_grid[i*get_num_y_points()+j] = get_y_grid_value(grid_j);
		    		if (is_border_point(grid_i, grid_j))
						gp_is_not_border[i*get_num_y_points()+j] = 0.0;
					else
						gp_is_not_border[i*get_num_y_points()+j] = 1.0;
					if ((i < get_num_x_points() - 1) && (j < get_num_y_points() - 1)) {
						gp_x_h_step[i*get_num_y_points()+j] = get_x_h_step(grid_i);
						gp_y_h_step[i*get_num_y_points()+j] = get_y_h_step(grid_j);
					}
		    	}
			}
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


__global__ void gpu_reduce_sum(double *input, double *results, int n) {
	extern __shared__ double sdata[];
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	// load input into __shared__ memory
	double x = 0.0;
	if(i < n)
		x = input[i];

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


double scalar_product(GridParameters gp, const double* f1, const double* f2) {
 	int size = gp.get_num_x_points() * gp.get_num_y_points() * sizeof(double);
    int numElements = gp.get_num_x_points() * gp.get_num_y_points();

	double *d_f1, *d_f2, *d_hxhy, *d_product;
    SAFE_CUDA(cudaMalloc(&d_f1, size));
    SAFE_CUDA(cudaMalloc(&d_f2, size));
    SAFE_CUDA(cudaMalloc(&d_hxhy, size));

    SAFE_CUDA(cudaMemcpy(d_f1, f1, size, cudaMemcpyHostToDevice));
    SAFE_CUDA(cudaMemcpy(d_f2, f2, size, cudaMemcpyHostToDevice));
    SAFE_CUDA(cudaMemcpy(d_hxhy, gp.hxhy, size, cudaMemcpyHostToDevice));

	int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	SAFE_CUDA(cudaMalloc(&d_product, (blocksPerGrid+1)*sizeof(double)));
    //printf("rank=%d CUDA kernel launch with %d blocks of %d threads numElements=%d\n", gp.rank, blocksPerGrid, threadsPerBlock, numElements);
    // reduce per-block partial sums 
    gpu_scalar_product<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_f1, d_f2, d_hxhy, d_product, numElements); CUDA_CHECK_ERROR;

    // reduce partial sums to a total sum
    gpu_reduce_sum<<<1, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_product, d_product+blocksPerGrid, blocksPerGrid); CUDA_CHECK_ERROR;

    double gpu_product = 0.0;
    SAFE_CUDA(cudaMemcpy(&gpu_product, d_product+blocksPerGrid, sizeof(double), cudaMemcpyDeviceToHost));
    //printf("rank=%d size=%d numElements=%d blocksPerGrid=%d CPU product=%f GPU product=%f\n", 
    //	gp.rank, size, numElements, blocksPerGrid, product, gpu_product);

    SAFE_CUDA(cudaFree(d_f1));
    SAFE_CUDA(cudaFree(d_f2));
    SAFE_CUDA(cudaFree(d_hxhy));
    SAFE_CUDA(cudaFree(d_product));

    double global_product = 0.0;
    int status = MPI_Allreduce(&gpu_product, &global_product, 1, MPI_DOUBLE, MPI_SUM, gp.comm);
    if (status != MPI_SUCCESS) throw std::runtime_error("Error in compute scalar_product!");
    //printf("rank %d: product=%f global_product=%f\n", gp.rank, product, global_product);
    return global_product;
}

void compute_delta(GridParameters gp, const double *func, double *delta_func, double f_top, double f_bottom, double f_left, double f_right, int i, int j, int grid_i, int grid_j) {
	double h_i_1 = gp.get_x_h_step(grid_i-1);
	double h_i = gp.get_x_h_step(grid_i);
	double h_j_1 = gp.get_y_h_step(grid_j-1);
	double h_j = gp.get_y_h_step(grid_j);
	double average_hx = (h_i + h_i_1) / 2.0;
	double average_hy = (h_j + h_j_1) / 2.0;
	double f_curr = func[i*gp.get_num_y_points()+j];
	delta_func[i*gp.get_num_y_points()+j] = 
		(1.0 / average_hx) * ((f_curr - f_top) / h_i_1 - (f_bottom - f_curr) / h_i) + 
		(1.0 / average_hy) * ((f_curr - f_left) / h_j_1 - (f_right - f_curr) / h_j);
	//printf("i=%d j=%d grid_i=%d grid_j=%d average_hx=%f average_hy=%f h_i_1=%f h_i=%f h_j_1=%f h_j=%f f_curr=%f f_top=%f f_bottom=%f f_left=%f f_right=%f delta_func[i][j] = %f\n", i, j, grid_i, grid_j, average_hx, average_hy, h_i_1, h_i, h_j_1, h_j, f_curr, f_top, f_bottom, f_left, f_right, delta_func[i*gp.get_num_y_points()+j]);
}

__global__ void gpu_compute_approx_delta(double *delta_func, double *func, double *gp_x_h_step, double* gp_y_h_step, 
	double* gp_is_not_border, int y_shape, int numElements) {
  int no_thread = threadIdx.x + blockDim.x * blockIdx.x;

  if ((no_thread < numElements) && (gp_is_not_border[no_thread] == 1.0)) {
  	double h_i_1 = gp_x_h_step[no_thread-1];
	double h_i = gp_x_h_step[no_thread];
	double h_j_1 = gp_y_h_step[no_thread-1];
	double h_j = gp_y_h_step[no_thread];
  	double average_hx = (h_i + h_i_1) / 2.0;
	double average_hy = (h_j + h_j_1) / 2.0;
	double f_curr = func[no_thread];
	double f_top = func[no_thread-y_shape];
	double f_bottom = func[no_thread+y_shape];
	double f_left = func[no_thread-1];
	double f_right = func[no_thread+1];
  	delta_func[no_thread] = (1.0 / average_hx) * ((f_curr - f_top) / h_i_1 - (f_bottom - f_curr) / h_i) + 
  			(1.0 / average_hy) * ((f_curr - f_left) / h_j_1 - (f_right - f_curr) / h_j);
  }
}

enum MPI_tags { SendToTop, SendToBottom, SendToLeft, SendToRight};

void compute_approx_delta(GridParameters gp, double* delta_func, const double* func) {
	int i, j;

	int size = gp.get_num_x_points() * gp.get_num_y_points() * sizeof(double);
    int numElements = gp.get_num_x_points() * gp.get_num_y_points();

	double *d_delta_func, *d_func, *d_gp_x_h_step, *d_gp_y_h_step, *d_gp_is_not_border;
	SAFE_CUDA(cudaMalloc(&d_delta_func, size));
	SAFE_CUDA(cudaMalloc(&d_func, size));
	SAFE_CUDA(cudaMalloc(&d_gp_x_h_step, size));
	SAFE_CUDA(cudaMalloc(&d_gp_y_h_step, size));
	SAFE_CUDA(cudaMalloc(&d_gp_is_not_border, size));

	SAFE_CUDA(cudaMemcpy(d_delta_func, delta_func, size, cudaMemcpyHostToDevice));
	SAFE_CUDA(cudaMemcpy(d_func, func, size, cudaMemcpyHostToDevice));
	SAFE_CUDA(cudaMemcpy(d_gp_x_h_step, gp.gp_x_h_step, size, cudaMemcpyHostToDevice));
	SAFE_CUDA(cudaMemcpy(d_gp_y_h_step, gp.gp_y_h_step, size, cudaMemcpyHostToDevice));
	SAFE_CUDA(cudaMemcpy(d_gp_is_not_border, gp.gp_is_not_border, size, cudaMemcpyHostToDevice));

	int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    gpu_compute_approx_delta<<<blocksPerGrid, threadsPerBlock>>>(d_delta_func, d_func, d_gp_x_h_step, d_gp_y_h_step, d_gp_is_not_border, gp.get_num_y_points(), numElements); CUDA_CHECK_ERROR;

    SAFE_CUDA(cudaMemcpy(delta_func, d_delta_func, size, cudaMemcpyDeviceToHost));

    SAFE_CUDA(cudaFree(d_delta_func));
    SAFE_CUDA(cudaFree(d_func));
    SAFE_CUDA(cudaFree(d_gp_x_h_step));
    SAFE_CUDA(cudaFree(d_gp_y_h_step));
    SAFE_CUDA(cudaFree(d_gp_is_not_border));

	if (gp.send_message_top == NULL)
		gp.send_message_top = new double [gp.get_num_y_points()];
	if (gp.send_message_bottom == NULL)
		gp.send_message_bottom = new double [gp.get_num_y_points()];
	if (gp.send_message_left == NULL)
		gp.send_message_left = new double [gp.get_num_x_points()];
	if (gp.send_message_right == NULL)
		gp.send_message_right = new double [gp.get_num_x_points()];

	if (gp.recv_message_top == NULL)
		gp.recv_message_top = new double [gp.get_num_y_points()];
	if (gp.recv_message_bottom == NULL)
		gp.recv_message_bottom = new double [gp.get_num_y_points()];
	if (gp.recv_message_left == NULL)
		gp.recv_message_left = new double [gp.get_num_x_points()];
	if (gp.recv_message_right == NULL)
		gp.recv_message_right = new double [gp.get_num_x_points()];

	if (gp.send_requests == NULL)
		gp.send_requests = new MPI_Request [4];
	if (gp.recv_requests == NULL)
		gp.recv_requests = new MPI_Request [4];

	for (int j=0; j<gp.get_num_y_points(); j++)
		gp.send_message_top[j] = func[0*gp.get_num_y_points()+j];
	for (int j=0; j<gp.get_num_y_points(); j++)
		gp.send_message_bottom[j] = func[(gp.get_num_x_points()-1)*gp.get_num_y_points()+j];
	for (int i=0; i<gp.get_num_x_points(); i++)
		gp.send_message_left[i] = func[i*gp.get_num_y_points()+0];
	for (int i=0; i<gp.get_num_x_points(); i++)
		gp.send_message_right[i] = func[i*gp.get_num_y_points()+gp.get_num_y_points()-1];

	int status;
	int send_count=0;
	if (not gp.top) {
		status = MPI_Isend(gp.send_message_top, gp.get_num_y_points(), MPI_DOUBLE, 
			gp.get_top_rank(), SendToTop, gp.comm, &(gp.send_requests[send_count]));
		if (status != MPI_SUCCESS) throw std::runtime_error("Error in send message!");
		send_count++;
	}
	if (not gp.bottom) {
		status = MPI_Isend(gp.send_message_bottom, gp.get_num_y_points(), MPI_DOUBLE, 
			gp.get_bottom_rank(), SendToBottom, gp.comm, &(gp.send_requests[send_count]));
		if (status != MPI_SUCCESS) throw std::runtime_error("Error in send message!");
		send_count++;
	}
	if (not gp.left) {
		status = MPI_Isend(gp.send_message_left, gp.get_num_x_points(), MPI_DOUBLE, 
			gp.get_left_rank(), SendToLeft, gp.comm, &(gp.send_requests[send_count]));
		if (status != MPI_SUCCESS) throw std::runtime_error("Error in send message!");
		send_count++;
	}
	if (not gp.right) {
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

    if (not gp.top) {
    	int i = 0;
    	for (int j=1; j<gp.get_num_y_points()-1; j++) {
    		int grid_i, grid_j;
    		gp.get_real_grid_index(i, j, grid_i, grid_j);
    		compute_delta(gp, func, delta_func, gp.recv_message_top[j], func[(i+1)*gp.get_num_y_points()+j], func[i*gp.get_num_y_points()+j-1], func[i*gp.get_num_y_points()+j+1], i, j, grid_i, grid_j);
    	}
    }

	if (not gp.bottom) {
    	int i = gp.get_num_x_points()-1;
    	for (int j=1; j<gp.get_num_y_points()-1; j++) {
    		int grid_i, grid_j;
    		gp.get_real_grid_index(i, j, grid_i, grid_j);
    		compute_delta(gp, func, delta_func, func[(i-1)*gp.get_num_y_points()+j], gp.recv_message_bottom[j], func[i*gp.get_num_y_points()+j-1], func[i*gp.get_num_y_points()+j+1], i, j, grid_i, grid_j);
    	}
    }

    if (not gp.left) {
    	int j = 0;
    	for (int i=1; i<gp.get_num_x_points()-1; i++) {
    		int grid_i, grid_j;
    		gp.get_real_grid_index(i, j, grid_i, grid_j);
    		compute_delta(gp, func, delta_func, func[(i-1)*gp.get_num_y_points()+j], func[(i+1)*gp.get_num_y_points()+j], gp.recv_message_left[i], func[i*gp.get_num_y_points()+j+1], i, j, grid_i, grid_j);
    	}
    }

    if (not gp.right) {
    	int j = gp.get_num_y_points()-1;
    	for (int i=1; i<gp.get_num_x_points()-1; i++) {
    		int grid_i, grid_j;
    		gp.get_real_grid_index(i, j, grid_i, grid_j);
    		compute_delta(gp, func, delta_func, func[(i-1)*gp.get_num_y_points()+j], func[(i+1)*gp.get_num_y_points()+j], func[i*gp.get_num_y_points()+j-1], gp.recv_message_right[i], i, j, grid_i, grid_j);
    	}
    }

    // compute corners
	i = 0; j = 0;
	if (not gp.top && not gp.left) {
		int grid_i, grid_j;
    	gp.get_real_grid_index(i, j, grid_i, grid_j);
    	compute_delta(gp, func, delta_func, gp.recv_message_top[j], func[(i+1)*gp.get_num_y_points()+j], gp.recv_message_left[i], func[i*gp.get_num_y_points()+j+1], i, j, grid_i, grid_j);
	}

	i = 0; j = gp.get_num_y_points()-1;
	if (not gp.top && not gp.right) {
		int grid_i, grid_j;
    	gp.get_real_grid_index(i, j, grid_i, grid_j);
    	compute_delta(gp, func, delta_func, gp.recv_message_top[j], func[(i+1)*gp.get_num_y_points()+j], func[i*gp.get_num_y_points()+j-1], gp.recv_message_right[i], i, j, grid_i, grid_j);
	}

	i = gp.get_num_x_points()-1; j = 0;
	if (not gp.bottom && not gp.left) {
		int grid_i, grid_j;
    	gp.get_real_grid_index(i, j, grid_i, grid_j);
    	compute_delta(gp, func, delta_func, func[(i-1)*gp.get_num_y_points()+j], gp.recv_message_bottom[j], gp.recv_message_left[i], func[i*gp.get_num_y_points()+j+1], i, j, grid_i, grid_j);
	}

	i = gp.get_num_x_points()-1; j = gp.get_num_y_points()-1;
	if (not gp.bottom && not gp.right) {
		int grid_i, grid_j;
    	gp.get_real_grid_index(i, j, grid_i, grid_j);
    	compute_delta(gp, func, delta_func, func[(i-1)*gp.get_num_y_points()+j], gp.recv_message_bottom[j], func[i*gp.get_num_y_points()+j-1], gp.recv_message_right[i], i, j, grid_i, grid_j);
	}
}

__global__ void gpu_compute_r(const double *delta_p, double *gp_x_grid, double *gp_y_grid, double* gp_is_not_border, double* r, int numElements) {
  int no_thread = threadIdx.x + blockDim.x * blockIdx.x;

  if (no_thread < numElements) {
  	r[no_thread] = (delta_p[no_thread] - gpu_F(gp_x_grid[no_thread], gp_y_grid[no_thread])) * gp_is_not_border[no_thread];
  }
}

void compute_r(GridParameters gp, double *r, const double *delta_p) {
	int size = gp.get_num_x_points() * gp.get_num_y_points() * sizeof(double);
    int numElements = gp.get_num_x_points() * gp.get_num_y_points();

	double *d_r, *d_delta_p, *d_gp_x_grid, *d_gp_y_grid, *d_gp_is_not_border;
	SAFE_CUDA(cudaMalloc(&d_r, size));
	SAFE_CUDA(cudaMalloc(&d_delta_p, size));
	SAFE_CUDA(cudaMalloc(&d_gp_x_grid, size));
	SAFE_CUDA(cudaMalloc(&d_gp_y_grid, size));
	SAFE_CUDA(cudaMalloc(&d_gp_is_not_border, size));

	SAFE_CUDA(cudaMemcpy(d_r, r, size, cudaMemcpyHostToDevice));
	SAFE_CUDA(cudaMemcpy(d_delta_p, delta_p, size, cudaMemcpyHostToDevice));
	SAFE_CUDA(cudaMemcpy(d_gp_x_grid, gp.gp_x_grid, size, cudaMemcpyHostToDevice));
	SAFE_CUDA(cudaMemcpy(d_gp_y_grid, gp.gp_y_grid, size, cudaMemcpyHostToDevice));
	SAFE_CUDA(cudaMemcpy(d_gp_is_not_border, gp.gp_is_not_border, size, cudaMemcpyHostToDevice));

	int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    gpu_compute_r<<<blocksPerGrid, threadsPerBlock>>>(d_delta_p, d_gp_x_grid, d_gp_y_grid, d_gp_is_not_border, d_r, numElements); CUDA_CHECK_ERROR;

    SAFE_CUDA(cudaMemcpy(r, d_r, size, cudaMemcpyDeviceToHost));

    SAFE_CUDA(cudaFree(d_r));
    SAFE_CUDA(cudaFree(d_delta_p));
    SAFE_CUDA(cudaFree(d_gp_x_grid));
    SAFE_CUDA(cudaFree(d_gp_y_grid));
    SAFE_CUDA(cudaFree(d_gp_is_not_border));
}

__global__ void gpu_compute_g(double *g, double *r, double alpha, int numElements) {
  int no_thread = threadIdx.x + blockDim.x * blockIdx.x;

  if (no_thread < numElements) {
  	g[no_thread] = r[no_thread] - alpha * g[no_thread];
  }
}

void compute_g(GridParameters gp, double *g, double *r, double alpha) {
	int size = gp.get_num_x_points() * gp.get_num_y_points() * sizeof(double);
    int numElements = gp.get_num_x_points() * gp.get_num_y_points();

	double *d_g, *d_r;
	SAFE_CUDA(cudaMalloc(&d_g, size));
	SAFE_CUDA(cudaMalloc(&d_r, size));

	SAFE_CUDA(cudaMemcpy(d_g, g, size, cudaMemcpyHostToDevice));
	SAFE_CUDA(cudaMemcpy(d_r, r, size, cudaMemcpyHostToDevice));

	int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    gpu_compute_g<<<blocksPerGrid, threadsPerBlock>>>(d_g, d_r, alpha, numElements); CUDA_CHECK_ERROR;

    SAFE_CUDA(cudaMemcpy(g, d_g, size, cudaMemcpyDeviceToHost));

    SAFE_CUDA(cudaFree(d_g));
    SAFE_CUDA(cudaFree(d_r));
}

__global__ void gpu_compute_p(double *p, double *p_prev, double *g, double tau, int numElements) {
  int no_thread = threadIdx.x + blockDim.x * blockIdx.x;

  if (no_thread < numElements) {
  	p[no_thread] = p_prev[no_thread] - tau * g[no_thread];
  }
}


void compute_p(GridParameters gp, double *p, double* p_prev, double *g, double tau) {
	int size = gp.get_num_x_points() * gp.get_num_y_points() * sizeof(double);
    int numElements = gp.get_num_x_points() * gp.get_num_y_points();

	double *d_p, *d_p_prev, *d_g;
	SAFE_CUDA(cudaMalloc(&d_p, size));
	SAFE_CUDA(cudaMalloc(&d_p_prev, size));
	SAFE_CUDA(cudaMalloc(&d_g, size));

	SAFE_CUDA(cudaMemcpy(d_p_prev, p_prev, size, cudaMemcpyHostToDevice));
	SAFE_CUDA(cudaMemcpy(d_g, g, size, cudaMemcpyHostToDevice));

	int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    gpu_compute_p<<<blocksPerGrid, threadsPerBlock>>>(d_p, d_p_prev, d_g,  tau, numElements); CUDA_CHECK_ERROR;

    SAFE_CUDA(cudaMemcpy(p, d_p, size, cudaMemcpyDeviceToHost));

    SAFE_CUDA(cudaFree(d_p));
    SAFE_CUDA(cudaFree(d_p_prev));
    SAFE_CUDA(cudaFree(d_g));
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


__global__ void gpu_reduce_max(double *input, double *results, int n) {
	extern __shared__ double sdata[];
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	// load input into __shared__ memory
	double x = 0.0;
	if(i < n)
		x = input[i];

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
	int size = gp.get_num_x_points() * gp.get_num_y_points() * sizeof(double);
    int numElements = gp.get_num_x_points() * gp.get_num_y_points();

	double *d_p, *d_p_prev, *d_norm;
	SAFE_CUDA(cudaMalloc(&d_p, size));
	SAFE_CUDA(cudaMalloc(&d_p_prev, size));

	SAFE_CUDA(cudaMemcpy(d_p, p, size, cudaMemcpyHostToDevice));
	SAFE_CUDA(cudaMemcpy(d_p_prev, p_prev, size, cudaMemcpyHostToDevice));

	int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	SAFE_CUDA(cudaMalloc(&d_norm, (blocksPerGrid+1)*sizeof(double)));
    //printf("rank=%d CUDA kernel launch with %d blocks of %d threads numElements=%d\n", gp.rank, blocksPerGrid, threadsPerBlock, numElements);
    // reduce per-block partial sums 
    gpu_norm<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_p, d_p_prev, d_norm, numElements); CUDA_CHECK_ERROR;

    // reduce partial sums to a total sum
    gpu_reduce_max<<<1, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_norm, d_norm+blocksPerGrid, blocksPerGrid); CUDA_CHECK_ERROR;

    double gpu_norm = 0.0;
    SAFE_CUDA(cudaMemcpy(&gpu_norm, d_norm+blocksPerGrid, sizeof(double), cudaMemcpyDeviceToHost));
    //printf("rank=%d size=%d numElements=%d blocksPerGrid=%d CPU product=%f GPU product=%f\n", 
    //	gp.rank, size, numElements, blocksPerGrid, product, gpu_product);

    SAFE_CUDA(cudaFree(d_p));
    SAFE_CUDA(cudaFree(d_p_prev));
    SAFE_CUDA(cudaFree(d_norm));

	double global_norm = 0.0;
	int status = MPI_Allreduce(&gpu_norm, &global_norm, 1, MPI_DOUBLE, MPI_MAX, gp.comm);
    if (status != MPI_SUCCESS) throw std::runtime_error("Error in compute scalar_product!");
    //printf("rank %d: norm=%f global_norm=%f\n", gp.rank, norm, global_norm);
    return global_norm;
}

void init_vector(GridParameters gp, double* func) {
	int i, j;

	for (i=0; i<gp.get_num_x_points(); i++) {
    	for (j=0; j<gp.get_num_y_points(); j++) {
    		int grid_i, grid_j;
    		gp.get_real_grid_index(i, j, grid_i, grid_j);
    		func[i*gp.get_num_y_points()+j] = 0.0;
		}
	}
}

void init_p_prev(GridParameters gp, double* p_prev) {
	int i, j;

	for (i=0; i<gp.get_num_x_points(); i++) {
    	for (j=0; j<gp.get_num_y_points(); j++) {
    		int grid_i, grid_j;
    		gp.get_real_grid_index(i, j, grid_i, grid_j);
    		if (not gp.is_border_point(grid_i, grid_j)) {
                p_prev[i*gp.get_num_y_points()+j] = 0.0;
            }
            else {
                p_prev[i*gp.get_num_y_points()+j] = phi(gp.get_x_grid_value(grid_i), gp.get_y_grid_value(grid_j));
            }
		}
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
	    
	    double* p = new double [gp.get_num_x_points() * gp.get_num_y_points()];
	    double* p_prev = new double [gp.get_num_x_points() * gp.get_num_y_points()];
	    double* g = new double [gp.get_num_x_points() * gp.get_num_y_points()];
	    double* r = new double [gp.get_num_x_points() * gp.get_num_y_points()];
	    double* delta_p = new double [gp.get_num_x_points() * gp.get_num_y_points()];
	    double* delta_r = new double [gp.get_num_x_points() * gp.get_num_y_points()];
	    double* delta_g = new double [gp.get_num_x_points() * gp.get_num_y_points()];
	    
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

	    double* phi_on_grid = new double [gp.get_num_x_points() * gp.get_num_y_points()];
	    for (int i=0; i<gp.get_num_x_points(); i++) {
	    	for (int j=0; j<gp.get_num_y_points(); j++) {
	    		int grid_i, grid_j;
	    		gp.get_real_grid_index(i, j, grid_i, grid_j);
	    		phi_on_grid[i*gp.get_num_y_points()+j] = phi(gp.get_x_grid_value(grid_i), gp.get_y_grid_value(grid_j));
	    	}
		}

	    int n_iter = 1;
	    while (true) {
	    	compute_approx_delta(gp, delta_p, p_prev);
	    	compute_r(gp, r, delta_p);

	    	if (n_iter > 1) {
	    		compute_approx_delta(gp, delta_r, r);
	    		scalar_product_delta_r_and_g = scalar_product(gp, delta_r, g);
	    		alpha = 1.0 * scalar_product_delta_r_and_g / scalar_product_delta_g_and_g;
	    	}

	    	if (n_iter > 1) 
	    		compute_g(gp, g, r, alpha);
	    	else 
            	swap(g, r);

            compute_approx_delta(gp, delta_g, g);
            if (n_iter > 1) {
            	scalar_product_r_and_g = scalar_product(gp, r, g);
            }
            else {
            	scalar_product_r_and_g = scalar_product(gp, g, g);
            }

            scalar_product_delta_g_and_g = scalar_product(gp, delta_g, g);
	        tau = 1.0 * scalar_product_r_and_g / scalar_product_delta_g_and_g;

	       	compute_p(gp, p, p_prev, g, tau);
	       	double norm_p_prev = compute_norm(gp, p, p_prev);
	       	double norm_p_phi = compute_norm(gp, p, phi_on_grid);
	       	if (rank == 0)
	       		printf("# iteration %d: norm_p_p_prev=%f norm_p_phi=%f\n", n_iter, norm_p_prev, norm_p_phi);
	       	if (norm_p_prev < gp.eps)
            	break;

            swap(p, p_prev);
	    	n_iter += 1;
	    }
	}
	MPI_Finalize();

	if (rank == 0) {
		clock_t end = clock();
	  	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	  	printf("Algorithm finished! Elapsed time: %f sec\n", elapsed_secs);
	}
	return 0;
}
