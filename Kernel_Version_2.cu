#include <stdio.h>

#include <stdlib.h>

#include "ppm_lib.h"


static void HandleError(cudaError_t err,
  const char * file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

# define HANDLE_ERROR(err)(HandleError(err, __FILE__, __LINE__))# define CREATOR "PARALLELISME2OPENMP"

PPMImage * readPPM(const char * filename) {
  char buff[16];
  PPMImage * img;
  FILE * fp;
  int c, rgb_comp_color;
  fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open file '%s'\n", filename);
    exit(1);
  }
  if (!fgets(buff, sizeof(buff), fp)) {
    perror(filename);
    exit(1);
  }
  if (buff[0] != 'P' || buff[1] != '6') {
    fprintf(stderr, "Invalid image format (must be 'P6')\n");
    exit(1);
  }
  img = (PPMImage * ) malloc(sizeof(PPMImage));
  if (!img) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }
  c = getc(fp);
  while (c == '#') {
    while (getc(fp) != '\n');
    c = getc(fp);
  }
  ungetc(c, fp);
  if (fscanf(fp, "%d %d", & img - > x, & img - > y) != 2) {
    fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
    exit(1);
  }
  if (fscanf(fp, "%d", & rgb_comp_color) != 1) {
    fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
    exit(1);
  }
  if (rgb_comp_color != RGB_COMPONENT_COLOR) {
    fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
    exit(1);
  }
  while (fgetc(fp) != '\n');
  img - > data = (PPMPixel * ) malloc(img - > x * img - > y * sizeof(PPMPixel));
  if (!img) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }
  if (fread(img - > data, sizeof(PPMPixel) * img - > x, img - > y, fp) != img - > y) {
    fprintf(stderr, "Error loading image '%s'\n", filename);
    exit(1);
  }
  fclose(fp);
  return img;
}
void writePPM(const char * filename, PPMImage * img) {
  FILE * fp;
  fp = fopen(filename, "wb");
  if (!fp) {
    fprintf(stderr, "Unable to open file '%s'\n", filename);
    exit(1);
  }
  fprintf(fp, "P6\n");
  fprintf(fp, "# Created by %s\n", CREATOR);
  fprintf(fp, "%d %d\n", img - > x, img - > y);
  fprintf(fp, "%d\n", RGB_COMPONENT_COLOR);
  fwrite(img - > data, 3 * img - > x, img - > y, fp);
  fclose(fp);
}

// Cuda version 2 (avec mémoire partagée)//
// Dans cette version  l'accés à l'image est deplacé de la mémoire globale (cad *img) à la mémoire partagée (en copiant les pixels voulu dans un tableau)
// Pour ca on a copié les pixels que chaque thread d'un block besoin à un tableau en mémoire partagé pour l'utiliser aprés.

__global__ void filterSofter(PPMPixel * img, int * filter, int & divisionFactor, PPMPixel * destination) {

  //un tableau de 3000 Pixel, dans notre cas 6 lignes de l'image.
  __shared__ PPMPixel final[3000];

  //Initialiser les Sommes de RVB pour les diviser sur le facteur du filtre.
  int finalRed = 0;
  int finalGreen = 0;
  int finalBlue = 0;
  // indice pour parcourir le filtre
  int indFiltre = 0;

  //indice du thread dans le block
  int tid = threadIdx.x;
  //indice du thread dans la grille, il nous servira a parcourir toute l'image puisque nbThreads=nbPixels.
  int tidX = threadIdx.x + blockIdx.x * blockDim.x;

  //Pour savoir dans quelle ligne de l'image se situe ce thread
  int ll = tidX / 500;
  //Pour savoir dans quelle colonne de l'image se situe ce thread
  int cc = tidX % 500;

  //c pour savoir dans quelle colonne du block se situe le thread 
  int c = tid % 500;
  int l;

  //chaque thread, dépendamment de sa position, soit il remplit le tableau final avec les deux pixels au dessus soit les deux au dessous
  //à la fin de ce calcul on aura les 6 lignes de l'image remplit dans le tableau final pour l'utiliser au lieu de l'image originale.
  if (tid < 500) {
    final[500 * 2 + c] = img[tidX];
    //Appliquer l'effet mirroir si on accéde a des indices supérieur ou inférieur à la taille de l'image
    if (ll - 1 >= 0)
      final[500 * 1 + c] = img[(ll - 1) * 500 + cc];
    else
      final[500 * 1 + c] = img[(ll + 1) * 500 + cc];

    if (ll - 2 >= 0)
      final[c] = img[(ll - 2) * 500 + cc];
    else
      final[c] = img[(ll + 2) * 500 + cc];

  } else {
    final[500 * 3 + c] = img[tidX];
    //Appliquer l'effet mirroir si on accéde a des indices supérieur ou inférieur à la taille de l'image
    if (ll + 1 > 1000)
      final[500 * 4 + c] = img[(ll + 1) * 500 + cc];
    else
      final[500 * 4 + c] = img[(ll - 1) * 500 + cc];

    if (ll + 2 > 1000)
      final[500 * 5 + c] = img[(ll + 2) * 500 + cc];
    else
      final[500 * 5 + c] = img[(ll - 2) * 500 + cc];

  }
  //pour s'assurer que tous les threads ont fini leurs travaille.
  __syncthreads();

  //on ajouter 1000 au tid pour repositionner l'indice du thread au bon endroit dans le tableau finale
  tid += 1000;
  l = tid / 500;
  c = tid % 500;

  //boucle pour parcourir tout les pixels autour
  for (int i = -2; i <= 2; i++) {
    for (int j = -2; j <= 2; j++) {
      ll = l + i;
      cc = c + j;
      //Appliquer l'effet mirroir si on accéde a des indices supérieur ou inférieur à la taille de l'image
      //cette fois on a pas besoin de tester avec les lignes parce qu'on a deja repositionner l'indice au milieu avec tid+=1000
      if (cc < 0) {
        cc = -cc;
      } else if (cc > 500) {
        cc = c - j;
      }

      //faire la somme des Pixel*Filtre
      finalRed += final[(ll) * 500 + (cc)].red * filter[indFiltre];
      finalGreen += final[(ll) * 500 + (cc)].green * filter[indFiltre];
      finalBlue += final[(ll) * 500 + (cc)].blue * filter[indFiltre];
      indFiltre++;

    }
  }

  //Affecter au pixel l'application du filtre
  destination[tidX].red = finalRed / divisionFactor;
  destination[tidX].green = finalGreen / divisionFactor;
  destination[tidX].blue = finalBlue / divisionFactor;

}

int main() {

  PPMImage * image, * imageCopy;
  image = readPPM("gare_parallelisme.ppm");
  imageCopy = readPPM("gare_parallelisme.ppm");

  /*int HorizontalSobel[25] = { 1,   2,   0,   -2,   -1,
                              4 ,  8,   0 ,  -8 ,  -4,
                              6  , 12 , 0 ,  -12  , -6 ,
                              4,   8,   0 ,  -8,    -4,
                              1,   2,   0,   -2,   -1 };

  int HorizontalSobelDivide=1;*/

  /*int VerticalSobel[25] = { -1,   -4,  -6,   -4,   -1,
                            -2 ,  -8,   -12 ,  -8 ,  -2,
                             0  , 0 , 0 ,  0  , 0 ,
                             2,   8,   12 ,  8,    2,
                             1,   4,   6,   4,   1 };

  int VerticalSobelDivide=1; */
  /*
  int DiagonalShatter[25] = { 1,   0,   0,   0,  1,
                             0 ,  0,   0 ,  0 ,  0,
                             0  , 0 , 0 ,  0  , 0 ,
                             0,   0,   0 ,  0,    0,
                             1,   0,   0,   0,   1 };

  int DiagonalShatterDivide=4;*/

  int HorizontalBlur[25] = {
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    2,
    3,
    2,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
  };

  int HorizontalBlurDivide = 9;
  /*
  int Soften[25] = { 1,   1,   1,   1,   1,
                     1 ,  1,   1 ,  1 ,  1,
                     1  , 1 , 1 ,  1  , 1 ,
                     1,   1,   1 ,  1,    1,
                     1,   1,   1,   1,   1 };
  int SoftenDivide=25;*/

  /*int SharpenMeduim[25] = { -1,   -1,   -1,   -1,  -1,
                            -1 ,  -1,   -1 ,  -1 , -1,
                            -1, -1, 49 , -1  , -1,
                            -1,   -1,   -1,  -1, -1,
                            -1,   -1,   -1,   -1, -1 };
  int SharpenMeduimDivide=25;
  */

  PPMPixel * dev_image;
  PPMPixel * dev_imageCopy;
  int * dev_filter;
  int * dev_divisionFactor;

  float time;
  cudaEvent_t start, stop;

  HANDLE_ERROR(cudaMalloc((void ** ) & dev_image, image - > x * image - > y * 3 * sizeof(char)));
  HANDLE_ERROR(cudaMalloc((void ** ) & dev_imageCopy, imageCopy - > x * imageCopy - > y * 3 * sizeof(char)));
  HANDLE_ERROR(cudaMalloc((void ** ) & dev_filter, 25 * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void ** ) & dev_divisionFactor, sizeof(int)));

  HANDLE_ERROR(cudaMemcpy(dev_image, image - > data, image - > x * image - > y * 3 * sizeof(char), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_imageCopy, imageCopy - > data, imageCopy - > x * imageCopy - > y * 3 * sizeof(char), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_filter, HorizontalBlur, 25 * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_divisionFactor, & HorizontalBlurDivide, sizeof(int), cudaMemcpyHostToDevice));

  cudaEventCreate( & start);
  cudaEventCreate( & stop);
  cudaEventRecord(start, 0);

  printf(">%s\n", cudaGetErrorString(cudaGetLastError()));

  for (int i = 0; i < 1000; i++) {
    filterSofter << < 500, 1000 >>> (dev_image, dev_filter, * dev_divisionFactor, dev_imageCopy);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime( & time, start, stop);

  HANDLE_ERROR(cudaMemcpy(imageCopy - > data, dev_imageCopy, imageCopy - > x * imageCopy - > y * 3 * sizeof(char), cudaMemcpyDeviceToHost));

  printf("Temps nécessaire :  %3.1f ms\n", time);

  writePPM("Result_Version_2.ppm", imageCopy);

  /* liberer la memoire allouee sur le GPU */
  cudaFree(dev_image);
  cudaFree(dev_imageCopy);
  cudaFree(dev_filter);

  return 0;
}