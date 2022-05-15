/*----------------------------------------------------------------------------*/
/*  FICHERO:       clasificacionSOM.cu									        */
/*  AUTOR:         Jorge Azorin								       			    */
/*													                            */
/*  RESUMEN												                        */
/*  ~~~~~~~												                        */
/* Ejercicio grupal para la clasificaci�n de patrones de entrada basada         */
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


__global__ void kernel(TSOM* d_SOM, TPatrones* d_Patrones, int* solucion_P) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	float distanciaMenor = MAXDIST;

	for (int h = 0; h < d_SOM->Alto; h++) {
		for (int a = 0; a < d_SOM->Ancho; a++) {
			float distancia = 0;
			for (int vy = -1;vy < 2;vy++)               // Calculo en la vecindad
				for (int vx = -1;vx < 2;vx++)
					if ((h + vy) >= 0 && (h + vy) < d_SOM->Alto && (a +vx) >= 0 && (a + vx) < d_SOM->Ancho)
					{
						for (int i = 0;i < d_Patrones->Dimension;i++)
							distancia += abs(d_SOM->Neurona[h][a].pesos[i] - d_Patrones->Pesos[tid][i]);
						distancia /= d_Patrones->Dimension;
					}
			if (distancia < distanciaMenor)
			{
				distanciaMenor = distancia;  // Neurona con menor distancia
				solucion_P[tid] = d_SOM->Neurona[h][a].label;
			}
		}
	}
}

/*----------------------------------------------------------------------------*/
/*  FUNCION A PARALELIZAR  (versi�n secuencial-CPU)  				          */
/*	Implementa la clasificaci�n basada en SOM de un conjunto de patrones      */
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
				distancia=CalculaDistancia(y,x,np);     // CalculaDistancia entre neurona (y,x) y patr�n np
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
	return OKCLAS;									// Simulaci�n CORRECTA
}

// ---------------------------------------------------------------
// ---------------------------------------------------------------
// FUNCION A IMPLEMENTAR POR EL GRUPO (paralelizaci�n de ClasificacionSOMCPU)
// ---------------------------------------------------------------
// ---------------------------------------------------------------

 int ClasificacionSOMGPU()
{
	TSOM* d_SOM;
	TNeurona** d_Neuronas;
	TNeurona** h_Neuronas;
	float** d_Patrones;
	float** h_Patrones;
	int* solucion_P;


	h_Neuronas = (TNeurona**)malloc(SOM.Alto * sizeof(TNeurona*));
	//Asignamos y copiamos el array de neuronas
	cudaMalloc(&d_Neuronas, SOM.Alto * sizeof(TNeurona*));

	for (int i = 0; i < SOM.Alto; i++) {
		cudaMalloc(&h_Neuronas[i], SOM.Ancho * sizeof(TNeurona));
		for (int j = 0; j < SOM.Ancho; j++) {
			cudaMemcpy(&h_Neuronas[i][j], &SOM.Neurona[i][j], sizeof(TNeurona), cudaMemcpyHostToDevice);
		}
	}
	cudaMemcpy(d_Neuronas, h_Neuronas, SOM.Alto * sizeof(TNeurona), cudaMemcpyHostToDevice);
	//Asignamos y copiamos los patrones
	h_Patrones = (float**)malloc(Patrones.Cantidad * sizeof(float*));
	cudaMalloc(&d_Patrones, Patrones.Cantidad * sizeof(float*));
	for (int i = 0; i < Patrones.Cantidad; i++) {
		cudaMalloc(&h_Patrones[i], Patrones.Dimension * sizeof(float));
		for (int j = 0; j < Patrones.Dimension; j++) {
			cudaMemcpy(&h_Patrones[i][j], &Patrones.Pesos[i][j], sizeof(float), cudaMemcpyHostToDevice);
		}
	}
	cudaMemcpy(d_Patrones, h_Patrones, Patrones.Cantidad * sizeof(float*), cudaMemcpyHostToDevice);
	//Asignamos espacio para la soluci�n
	cudaMalloc(&solucion_P, Patrones.Cantidad * sizeof(int));


	cudaMemcpy(EtiquetaGPU, &solucion_P, (Patrones.Cantidad * sizeof(int)), cudaMemcpyDeviceToHost);

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
	
	// Creaci�n etiquetas resultados para versiones CPU y GPU

	EtiquetaCPU = (int*)malloc(Patrones.Cantidad*sizeof(int));
	EtiquetaGPU = (int*)malloc(Patrones.Cantidad*sizeof(int));
	
	/* Algoritmo a paralelizar */
	cpu_start_time = getTime();
	if (ClasificacionSOMCPU() == ERRORCLASS)
	{
		fprintf(stderr, "Clasificaci�n CPU incorrecta\n");
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
		fprintf(stderr, "Clasificaci�n GPU incorrecta\n");
		BorrarMapa();
		if (EtiquetaCPU != NULL) free(EtiquetaCPU);
		if (EtiquetaGPU != NULL) free(EtiquetaGPU);
		return;
	}
	cudaDeviceSynchronize();
	gpu_end_time = getTime();
	// Comparaci�n de correcci�n
	int comprobar = OKCLAS;
	for (int i = 0; i<Patrones.Cantidad; i++)
	{
		if ((EtiquetaCPU[i] != EtiquetaGPU[i]))
		{
			comprobar = ERRORCLASS;
			fprintf(stderr, "Fallo en la clasificacion del patron %d, valor correcto %d\n", i, EtiquetaCPU[i]);
		}
	}
	// Impresion de resultados
	if (comprobar == OKCLAS)
	{
		printf("Clasificacion correcta!\n");

	}
	// Impresi�n de resultados
	printf("Tiempo ejecuci�n GPU : %fs\n", \
		gpu_end_time - gpu_start_time);
	printf("Tiempo de ejecuci�n en la CPU : %fs\n", \
		cpu_end_time - cpu_start_time);
	printf("Se ha conseguido un factor de aceleraci�n %fx utilizando CUDA\n", (cpu_end_time - cpu_start_time) / (gpu_end_time - gpu_start_time));
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
/*	Funci�n:  LeerSOM(char *fichero)						              */
/*													                          */
/*	          Lee la estructura del SOM con formato .SOM   */
/*----------------------------------------------------------------------------*/
int LeerSOM(const char *fichero)
{
	int i, j, ndim, count;		/* Variables de bucle */
	int alto,ancho;		/* Variables de tama�o del mapa */
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
/*	Funci�n:  LeerPatrones(char *fichero)						              */
/*													                          */
/*	          Lee los patrones de un fichero de entrada .pat   */
/*----------------------------------------------------------------------------*/
int LeerPatrones(const char *fichero)
{
	int i, ndim, count;		/* Variables de bucle */
	int cantidad,dimension;		/* Variables de tama�o de los patrones */
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

