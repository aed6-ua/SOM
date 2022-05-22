/*----------------------------------------------------------------------------*/
/*  FICHERO:       clasificacionSOM.cu									        */
/*  AUTOR:         Jorge Azorin								       			    */
/*													                            */
/*  RESUMEN												                        */
/*  ~~~~~~~												                        */
/* Ejercicio grupal para la clasificación de patrones de entrada basada         */
/* en SOM utilizando GPUs                                                       */
/*----------------------------------------------------------------------------*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>


// includes, project
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "clasificacionSOM.h"
#include <Windows.h>



#define ERROR_CHECK { cudaError_t err; if ((err = cudaGetLastError()) != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__);}}

typedef LARGE_INTEGER timeStamp;
double getTime();


__device__ float CalculaDistanciaGPU(int y, int x, int np, TNeurona** d_Neuronas, float** d_Patrones, int dimension, int alto, int ancho) {
	float distancia = 0;
	if (y >= 0 && y < alto && x >= 0 && x < ancho)
	{
		for (int i = 0;i < dimension;i++)
			distancia += fabs(d_Neuronas[y][x].pesos[i] - d_Patrones[np][i]);
		distancia /= dimension;
	}
	return distancia;
}

__global__ void kernel(TNeurona** d_Neuronas, float** d_Patrones, int* solucion_P, int alto, int ancho, int dimension, int cantidad) {
	const int bid = blockIdx.x; //Cada bloque es un patrón
	int j = (threadIdx.y * blockDim.x) + threadIdx.x; //Id del thread
	extern __shared__ int distanciaYLabels[]; //Declaro un solo dynamic shared array
	float* distancia = (float*)distanciaYLabels;
	int* neuronaLabel = &distanciaYLabels[alto * ancho];
	float distanciaMenor = MAXDIST;
	float dist;
	int vx, vy;
		//Solución de la cpu tal cual (menos recorrer los patrones ya que cada bloque es un patrón)
		//for (int y = 0; y < alto; y++) {
			//for (int x = 0; x < ancho; x++) {

				neuronaLabel[j] = d_Neuronas[threadIdx.y][threadIdx.x].label;
				distancia[j] = CalculaDistanciaGPU(threadIdx.y, threadIdx.x, bid, d_Neuronas, d_Patrones, dimension, alto, ancho);


				for (vy = -1;vy < 2;vy++)               // Calculo en la vecindad
					for (vx = -1;vx < 2;vx++)
					{
						if (vx != 0 && vy != 0) {


							distancia[j] += CalculaDistanciaGPU(threadIdx.y + vy, threadIdx.x + vx, bid, d_Neuronas, d_Patrones, dimension, alto, ancho);
						}
					}
				int nmed = (alto * ancho) >> 1;
				int nelem = (alto * ancho);
				
				__syncthreads();
				for (unsigned int s = nmed; s > 0; s >>= 1) {
					if (j < s) {
						if (distancia[j] > distancia[s + j]) {
							distancia[j] = distancia[s + j];
							neuronaLabel[j] = neuronaLabel[s + j];
						}
						if ((nelem & 1) && (j == s - 1)) {
							if (distancia[j] > distancia[s << 1]) {
								distancia[j] = distancia[s << 1];
								neuronaLabel[j] = neuronaLabel[s << 1];
							}
						}
					}
					nelem = s;
					__syncthreads();
				}
				if (j == 0) {
					solucion_P[bid] = neuronaLabel[0];
				}
				/*if (threadIdx.y == 0 && threadIdx.x == 0)
				{
					for (int y = 0; y < alto; y++) {
						for (int x = 0; x < ancho; x++) {
							if (distancia[(y * ancho) + x] < distanciaMenor) {
								distanciaMenor = distancia[(y * ancho) + x];  // Neurona con menor distancia
								solucion_P[bid] = d_Neuronas[y][x].label;
							}
						}
					}
					
				}*/
			//}
		//}
	/*
	for (int i = 0; i < alto; i++) {
		for (int j = 0; j < ancho; j++) {
			solucion_P[(i * ancho) + j] = d_Neuronas[i][j].label;
		}
	}*/
}

/*----------------------------------------------------------------------------*/
/*  FUNCION A PARALELIZAR  (versión secuencial-CPU)  				          */
/*	Implementa la clasificación basada en SOM de un conjunto de patrones      */
/*  de entrada definidos en un fichero                                         */
/*----------------------------------------------------------------------------*/
int ClasificacionSOMCPU()
{
	float distancia;
	float distanciaMenor=MAXDIST;

	for (int np=0;np<Patrones.Cantidad;np++) // Recorrido de todos los patrones
	{
		distanciaMenor=MAXDIST;
		for (int y = 0; y<SOM.Alto; y++)			// Recorrido de todas las neuronas
		{
			for (int x = 0; x<SOM.Ancho; x++)
			{
				distancia=CalculaDistancia(y,x,np);     // CalculaDistancia entre neurona (y,x) y patrón np
				for (int vy=-1;vy<2;vy++)               // Calculo en la vecindad
					for (int vx=-1;vx<2;vx++)
						if (vx != 0 && vy != 0)         // No comprobar con la misma neurona
						   distancia+=CalculaDistancia(y+vy,x+vx,np);
				if (distancia < distanciaMenor)
				{
					distanciaMenor=distancia;  // Neurona con menor distancia
					EtiquetaCPU[np]=SOM.Neurona[y][x].label;
				}

			}
		}

	}
	return OKCLAS;									// Simulación CORRECTA
}

// ---------------------------------------------------------------
// ---------------------------------------------------------------
// FUNCION A IMPLEMENTAR POR EL GRUPO (paralelización de ClasificacionSOMCPU)
// ---------------------------------------------------------------
// ---------------------------------------------------------------

 int ClasificacionSOMGPU()
{
	 TNeurona** d_SOM;
	 
	 int* d_Solucion;
	 float** d_Patrones;

	 //Asignar espacio para los punteros a filas de neuronas
	 ERROR_CHECK(cudaMalloc((void**)&d_SOM, SOM.Alto * sizeof(TNeurona*)));
	 //vector de punteros temporal de filas de neuronas
	 TNeurona** temp_d_som = (TNeurona**)malloc(sizeof(TNeurona*) * SOM.Alto);

	 for (int j = 0; j < SOM.Alto; j++) {

		 //asignar espacio para cada fila de neuronas y guardar el puntero en el vector temporal
		 ERROR_CHECK(cudaMalloc((void**)&temp_d_som[j], SOM.Ancho * sizeof(TNeurona)));
		 //fila de neuronas temporal
		 TNeurona* temp_d_ne = (TNeurona*)malloc(sizeof(TNeurona) * SOM.Ancho);


		 for (int i = 0; i < SOM.Ancho; i++) {

			 //asignar espacio para cada vector de pesos de las neuronas
			 ERROR_CHECK(cudaMalloc((void**)& temp_d_ne[i].pesos, SOM.Dimension * sizeof(float)));
			 //copiar el vector de pesos al vector de neuronas temporal
			 ERROR_CHECK(cudaMemcpy(temp_d_ne[i].pesos, SOM.Neurona[j][i].pesos, SOM.Dimension * sizeof(float), cudaMemcpyHostToDevice));
			 //copiar el label
			 temp_d_ne[i].label = SOM.Neurona[j][i].label;
		 }

		 //copiar cada puntero a una fila de neuronas al vector de punteros a filas
		 ERROR_CHECK(cudaMemcpy(temp_d_som[j], temp_d_ne, SOM.Ancho * sizeof(TNeurona), cudaMemcpyHostToDevice));
		 free(temp_d_ne);
	 }
	 //copiar el vector de punteros a filas a la gpu
	 ERROR_CHECK(cudaMemcpy(d_SOM, temp_d_som, SOM.Alto * sizeof(TNeurona*), cudaMemcpyHostToDevice));
	 free(temp_d_som);

	 //asignar espacio al vector solución
	 ERROR_CHECK(cudaMalloc((void**)&d_Solucion, Patrones.Cantidad * sizeof(int)));
	 

	 //asignar espacio al vector de patrones
	 ERROR_CHECK(cudaMalloc((void**) & d_Patrones, Patrones.Cantidad * sizeof(float*)));
	 //crear vector de patrones temporal
	 float** temp_d_ptrs = (float**)malloc(sizeof(float*) * Patrones.Cantidad);

	 for (int i = 0; i < Patrones.Cantidad; i++) {

		 //asignar espacio para cada patrón (vector de pesos)
		 ERROR_CHECK(cudaMalloc((void**) &temp_d_ptrs[i], Patrones.Dimension * sizeof(float)));
		 //copiar cada patrón
		 ERROR_CHECK(cudaMemcpy(temp_d_ptrs[i], Patrones.Pesos[i], Patrones.Dimension * sizeof(float), cudaMemcpyHostToDevice));
	 }

	 //copiar el vector de patrones a la gpu
	 ERROR_CHECK(cudaMemcpy(d_Patrones, temp_d_ptrs, sizeof(float*) * Patrones.Cantidad, cudaMemcpyHostToDevice));
	 free(temp_d_ptrs);
	 
	 

	 dim3 block(SOM.Alto, SOM.Ancho);
	 dim3 grid(Patrones.Cantidad);

	 kernel <<<grid, block, (SOM.Alto * SOM.Ancho * sizeof(float) + (SOM.Alto * SOM.Ancho * sizeof(int))) >> > (d_SOM, d_Patrones, d_Solucion, SOM.Alto, SOM.Ancho, Patrones.Dimension, Patrones.Cantidad);


	 //copiar la solucion de la gpu
	 ERROR_CHECK(cudaMemcpy(EtiquetaGPU, d_Solucion, (Patrones.Cantidad * sizeof(int)), cudaMemcpyDeviceToHost));
	 cudaFree(d_SOM);
	 cudaFree(d_Patrones);
	 cudaFree(d_Solucion);

	 return OKCLAS;
}
 // ---------------------------------------------------------------
 // ---------------------------------------------------------------
 // ---------------------------------------------------------------
 // ---------------------------------------------------------------
 // ---------------------------------------------------------------

 // Declaraciones adelantadas de funciones
 int LeerSOM(const char *fichero);
 int LeerPatrones(const char *fichero);
 


////////////////////////////////////////////////////////////////////////////////
//PROGRAMA PRINCIPAL
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char** argv)
{

  	double gpu_start_time, gpu_end_time;
	double cpu_start_time, cpu_end_time;

	/* Numero de argumentos */
	if (argc != 3)
	{
		fprintf(stderr, "Numero de parametros incorecto\n");
		fprintf(stderr, "Uso: %s estructura.som patrones.pat\n", argv[0]);
		return;
	}

	/* Apertura de Fichero */
	printf("Clasificacion basada en SOM...\n");
	/* Mapa SOM */
	if (LeerSOM((char *)argv[1]) == ERRORCLASS)
	{
		fprintf(stderr, "Lectura de SOM incorrecta\n");
		return;
	}
	/* Patrones */
	if (LeerPatrones((char *)argv[2]) == ERRORCLASS)
	{
		fprintf(stderr, "Lectura de patrones incorrecta\n");
		return;
	}
	
	// Creación etiquetas resultados para versiones CPU y GPU

	EtiquetaCPU = (int*)malloc(Patrones.Cantidad *sizeof(int));
	EtiquetaGPU = (int*)malloc(Patrones.Cantidad*sizeof(int));
	
	/* Algoritmo a paralelizar */
	cpu_start_time = getTime();
	if (ClasificacionSOMCPU() == ERRORCLASS)
	{
		fprintf(stderr, "Clasificación CPU incorrecta\n");
		BorrarMapa();
		if (EtiquetaCPU != NULL) free(EtiquetaCPU);
		if (EtiquetaGPU != NULL) free(EtiquetaCPU);
		exit(1);
	}
	cpu_end_time = getTime();
	cudaSetDevice(0);
	/* Algoritmo a implementar */
	gpu_start_time = getTime();
	if (ClasificacionSOMGPU() == ERRORCLASS)
	{
		fprintf(stderr, "Clasificación GPU incorrecta\n");
		BorrarMapa();
		if (EtiquetaCPU != NULL) free(EtiquetaCPU);
		if (EtiquetaGPU != NULL) free(EtiquetaGPU);
		return;
	}
	cudaDeviceSynchronize();
	gpu_end_time = getTime();
	// Comparación de corrección
	int comprobar = OKCLAS;
	/*for (int i = 0; i < SOM.Alto;i++) {
		for (int j = 0;j < SOM.Ancho;j++) {
			if (EtiquetaGPU[(i * SOM.Ancho) + j] != SOM.Neurona[i][j].label)
				printf("Neurona mal copiada\n");
		}
	}*/
	for (int i = 0; i<Patrones.Cantidad; i++)
	{
		if ((EtiquetaCPU[i] != EtiquetaGPU[i]))
		{
			comprobar = ERRORCLASS;
			fprintf(stderr, "Fallo en la clasificacion del patron %d, valor correcto %d, valor dado %d\n", i, EtiquetaCPU[i], EtiquetaGPU[i]);
		}
	}
	// Impresion de resultados
	if (comprobar == OKCLAS)
	{
		printf("Clasificacion correcta!\n");

	}
	// Impresión de resultados
	printf("Tiempo ejecución GPU : %fs\n", \
		gpu_end_time - gpu_start_time);
	printf("Tiempo de ejecución en la CPU : %fs\n", \
		cpu_end_time - cpu_start_time);
	printf("Se ha conseguido un factor de aceleración %fx utilizando CUDA\n", (cpu_end_time - cpu_start_time) / (gpu_end_time - gpu_start_time));
	// Limpieza de Neuronas
	BorrarMapa();
	BorrarPatrones();
	if (EtiquetaCPU != NULL) free(EtiquetaCPU);
	if (EtiquetaGPU != NULL) free(EtiquetaGPU);
	return;
}

int
main(int argc, char** argv)
{
	runTest(argc, argv);
	getchar();
}

/* Funciones auxiliares */
double getTime()
{
	timeStamp start;
	timeStamp dwFreq;
	QueryPerformanceFrequency(&dwFreq);
	QueryPerformanceCounter(&start);
	return double(start.QuadPart) / double(dwFreq.QuadPart);
}



/*----------------------------------------------------------------------------*/
/*	Función:  LeerSOM(char *fichero)						              */
/*													                          */
/*	          Lee la estructura del SOM con formato .SOM   */
/*----------------------------------------------------------------------------*/
int LeerSOM(const char *fichero)
{
	int i, j, ndim, count;		/* Variables de bucle */
	int alto,ancho;		/* Variables de tamaño del mapa */
	FILE *fpin; 			/* Fichero */
	int nx,ny,lx,ly,label,dimension;
	float pesos;

	/* Apertura de Fichero */
	if ((fpin = fopen(fichero, "r")) == NULL) return ERRORCLASS;
	/* Lectura de cabecera */
	if (fscanf(fpin, "Alto: %d\n", &alto)<0) return ERRORCLASS;
	if (fscanf(fpin, "Ancho: %d\n", &ancho)<0) return ERRORCLASS;
	if (fscanf(fpin, "Dimension: %d\n", &dimension)<0) return ERRORCLASS;
	if (feof(fpin)) return ERRORCLASS;

	if (CrearMapa(alto, ancho, dimension) == ERRORCLASS) return ERRORCLASS;
	/* Lectura del SOM */
	count = 0;
	for (i = 0; i<ancho; i++)
	{
		for (j = 0; j<alto; j++)
		{
			if (!feof(fpin))
			{
				fscanf(fpin, "N%d,%d:", &nx, &ny);
				for (ndim = 0; ndim<dimension-1;ndim++)
				{
			        fscanf(fpin, " %f", &pesos);
				    SOM.Neurona[j][i].pesos[ndim] = pesos;
				}
				fscanf(fpin, " %f\n", &pesos);
				SOM.Neurona[j][i].pesos[ndim] = pesos;
				
				fscanf(fpin, "L%d,%d: %d\n", &lx, &ly, &label);
				SOM.Neurona[j][i].label=label;
			
				count++;
			}
			else break;
		}
	}
	fclose(fpin);
	if (count != ancho*alto) return ERRORCLASS;
	return OKCLAS;
}

/*----------------------------------------------------------------------------*/
/*	Función:  LeerPatrones(char *fichero)						              */
/*													                          */
/*	          Lee los patrones de un fichero de entrada .pat   */
/*----------------------------------------------------------------------------*/
int LeerPatrones(const char *fichero)
{
	int i, ndim, count;		/* Variables de bucle */
	int cantidad,dimension;		/* Variables de tamaño de los patrones */
	FILE *fpin; 			/* Fichero */

	int np;
	float pesos;

	/* Apertura de Fichero */
	if ((fpin = fopen(fichero, "r")) == NULL) return ERRORCLASS;
	/* Lectura de cabecera */
	if (fscanf(fpin, "Numero: %d\n", &cantidad)<0) return ERRORCLASS;
	if (fscanf(fpin, "Dimension: %d\n", &dimension)<0) return ERRORCLASS;
	if (feof(fpin)) return ERRORCLASS;
	
	if (CrearPatrones(cantidad, dimension) == ERRORCLASS) return ERRORCLASS;
	/* Lectura de patrones */
	count = 0;
	for (i = 0; i<cantidad; i++)
	{
	  	if (!feof(fpin))
		{
			fscanf(fpin, "P%d:", &np);
				for (ndim = 0; ndim<dimension-1;ndim++)
				{
			        fscanf(fpin, " %f", &pesos);
					Patrones.Pesos[i][ndim] = pesos;
				}
				fscanf(fpin, " %f\n", &pesos);
				Patrones.Pesos[i][ndim] = pesos;
				
					
				count++;
			}
			else break;
	}
	
	fclose(fpin);
	if (count != cantidad) return ERRORCLASS;
	return OKCLAS;
}

