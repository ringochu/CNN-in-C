#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int band_size = 22, inputFrame = 9;  // column ~ InputFrame
int kenal_width = 3;
int outputFrame = 20;

//printf("\n %i \n", weightA[0][2] );

int main(){

    //Initalise matrix
    int matrixA[band_size][inputFrame]; // matrixA[7][3]
    int weightA[outputFrame][outputFrame*kenal_width]; //weightA[1][3]
    int output[band_size-kenal_width+1][outputFrame]; //output[7][1]

    // Output matrix = [1,2,3],[3,4,5]...[19,20,21]  with shape 7x3
    int *element;
    element = (int *)malloc(sizeof(int));
    *element = 1;

    printf("\nInput Matrics:\n");
    for(int i=0; i<band_size; i++){
        for(int j=0; j<inputFrame ; j++){
            matrixA[i][j] = *element;
            printf(" %d ", *element);
            ++(*element);
        }
        printf("\n");
    }
    free(element);


    //Ininitalize Weight element
    int intermediate_dimension = inputFrame * kenal_width;
    printf("network weights:\n");
    for(int i=0; i<outputFrame; i++){
        for(int j=0; j < intermediate_dimension; j++){
            weightA[i][j] = 10+i;
            printf(" %i ", weightA[i][j]);
        }
        printf("\n");
    }
    

    //Implement 1d convNet in Torch style on C
    int testVal = 0;
    //Output image height = originalSize - kenal width + kenal height(1) 
    int out_height = band_size-kenal_width+1; // 22-3+1 or 7-1+1

    for(int i=0 ;i < out_height; i++ ){    // Output row
        for(int j=0; j< outputFrame; j++){   //Out put Column
                //--printf("%i,%i\n" , i,j);
                //Calcuate dot product
                /* int counter = 0;  //Index for location the weight matricx
                    int dot_process = 0; //Intermediate step of calculating dot product
            
                    for(int k = i; k<kenal_width+i; k++){  
                    
                        for(int weight_length =0; weight_length < inputFrame; weight_length++){
                        dot_process += matrixA[k][weight_length]*weightA[][counter];

                        }
                    } */
            int counter = 0; //Intermediate weight
            int dot_process = 0; //Intermediate dot product calculation
            for(int k_w = i; k_w < i + kenal_width; k_w++ ){
                for(int k_len = 0; k_len< inputFrame; k_len++){
                    dot_process += matrixA[k_w][k_len] * weightA[j][counter];
                }
            }

            dot_process = dot_process + 0 ;//bias[]
            printf("dot_process : %i\n" , dot_process);
            output[i][j] = dot_process;
            ++testVal;
        }
    }

    //Debug
    printf("Random access element: %i", output[1][2]);
    printf("\n Test value, number of element :%i\n", testVal);
return 0;
}
