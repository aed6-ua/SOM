/*----------------------------------------------------------------------------*/
/*  FICHERO:       clasificacionSOM.h									          */
/*  AUTOR:         Jorge Azorin											  */
/*													                          */
/*  RESUMEN												                      */
/*  ~~~~~~~												                      */
/* Fichero de definiciones y estructuras                                      */
/*    						                                                  */
/*----------------------------------------------------------------------------*/

#ifndef _CLASIFICASOM_H_
#define _CLASIFICASOM_H_

/*============================================================================ */
/* Constantes											                       */
/*============================================================================ */
#define ERRORCLASS 1
#define OKCLAS    0
#define MAXDIST 10000.0

/*============================================================================ */
/* Estructuras											                       */
/*============================================================================ */

	struct sTNeurona
	{
		float* pesos;
		int label;

	};
	typedef struct sTNeurona TNeurona;

	
	struct sTSOM
	{
		int Ancho;
		int Alto;
		int Dimension;
		TNeurona** Neurona;
	};
	typedef struct sTSOM TSOM;

	struct sTPatrones
	{
		int Cantidad;
		int Dimension;
		float ** Pesos;
	};
	typedef struct sTPatrones TPatrones;

	
	/*============================================================================ */
	/* Variables Globales										                   */
	/*============================================================================ */
	TSOM SOM;
	TPatrones Patrones;

	int* EtiquetaCPU;
	int* EtiquetaGPU;
		
	/*============================================================================ */
	/* Funciones de tratamiento de memoria							 */
	/*============================================================================ */
	void BorrarMapa(void)
	{
		int i;
		if (SOM.Neurona != NULL)
		{
			for (i = 0; i < SOM.Ancho; i++)
			if (SOM.Neurona[i] != NULL) free(SOM.Neurona[i]);
			free(SOM.Neurona);
			SOM.Neurona = NULL;
		}
	}

	void BorrarPatrones(void)
	{
		int i;
		if (Patrones.Pesos != NULL)
		{
			for (i = 0; i < Patrones.Cantidad; i++)
			if (Patrones.Pesos[i] != NULL) free(Patrones.Pesos[i]);
			free(Patrones.Pesos);
			Patrones.Pesos = NULL;
		}
	}

	int CrearMapa(int Alto, int Ancho, int Dimension)
	{
		int i,j;
		SOM.Alto = Alto;
		SOM.Ancho = Ancho;
		SOM.Dimension = Dimension;
		SOM.Neurona = (TNeurona**)malloc(SOM.Alto*sizeof(TNeurona*));
		if (SOM.Neurona == NULL) return ERRORCLASS;
		for (j = 0; j < SOM.Alto; j++)
		{
			SOM.Neurona[j] = (TNeurona*)malloc(SOM.Ancho*(int)sizeof(TNeurona));
			if (SOM.Neurona[j] == NULL)
			{
				BorrarMapa();
				return ERRORCLASS;
			}
			for (i = 0; i < SOM.Ancho; i++)
				SOM.Neurona[j][i].pesos = (float*)malloc(Dimension*(int)sizeof(float));
		}
		return OKCLAS;
	}

	int CrearPatrones(int Cantidad, int Dimension)
	{
		int j;
		Patrones.Cantidad=Cantidad;
		Patrones.Dimension=Dimension;
		Patrones.Pesos = (float**)malloc(Cantidad*sizeof(float*));
		for (j = 0; j < Cantidad; j++)
		{
			Patrones.Pesos[j] = (float*)malloc(Dimension*(int)sizeof(float));
			if (Patrones.Pesos[j] == NULL)
			{
				BorrarPatrones();
				return ERRORCLASS;
			}
			
		}
		return OKCLAS;
	}
	float CalculaDistancia(int y, int x, int np)
	{
		float distancia=0;
		if (y>=0 && y<SOM.Alto && x>=0 && x<SOM.Ancho)
		{
		for (int i=0;i<Patrones.Dimension;i++)
			distancia+=fabs(SOM.Neurona[y][x].pesos[i]-Patrones.Pesos[np][i]);
		distancia/=Patrones.Dimension;
		}
		return distancia;
	}


#endif // _CLASIFICASOM_H_

