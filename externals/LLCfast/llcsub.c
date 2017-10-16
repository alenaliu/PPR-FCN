#include <string.h>
#include <math.h>
#include <mex.h>

void inv(double *a, double *b,int n){
    double tem=0,temp=0,temp1=0,temp2=0,temp4=0,temp5=0;
	int m=0,i=0,j=0,p=0,q=0;
 	
	for(i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
		{
			if(i==j)
			b[j*n+i]=1.0;
			else
			b[j*n+i]=0.0;
		}
	}
	for(i=0;i<n;i++)
	{
		temp=a[i*n+i];
		if(temp<0)
		temp=temp*(-1);
		p=i;
		for(j=i+1;j<n;j++)
		{
			if(a[i*n+j]<0)
				tem=a[i*n+j]*(-1);
			else
				tem=a[i*n+j];
			if(temp<0)
				temp=temp*(-1);
			if(tem>temp)
			{
				p=j;
				temp=a[i*n+j];
			}
		}
		/*row exchange in both the matrix*/
		for(j=0;j<n;j++)
		{
			temp1=a[j*n+i];
			a[j*n+i]=a[j*n+p];
			a[j*n+p]=temp1;
			temp2=b[j*n+i];
			b[j*n+i]=b[j*n+p];
			b[j*n+p]=temp2;
		}
		/*dividing the row by a[i][i]*/
		temp4=a[i*n+i];
		for(j=0;j<n;j++)
		{
			a[j*n+i]=(double)a[j*n+i]/temp4;
			b[j*n+i]=(double)b[j*n+i]/temp4;
		}
		/*making other elements 0 in order to make the matrix a[][] an indentity matrix and obtaining a inverse b[][] matrix*/
		for(q=0;q<n;q++)
		{
			if(q==i)
				continue;
			temp5=a[i*n+q];
			for(j=0;j<n;j++)
			{
				a[j*n+q]=a[j*n+q]-(temp5*a[j*n+i]);
				b[j*n+q]=b[j*n+q]-(temp5*b[j*n+i]);
			}
		}
	}
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    /*input variables*/
    double *IDX;/*nframes(N) x knn*/
    double *B;/*ncodebook(K) x nfeatures(D)*/
    double *X;/*nframes(N) x D*/
    double beta;
    /*output variables*/
    double *C;/*codes, nframes(N) x ncodebook(K)*/
    /*dimensions*/
    mwSize knn,N,K,D;

    /*temp varaible and iterators*/
    double *z,*c,*cinv,*w;
    int* idx;
   
    mwSize ii,iii,j,jj,i;
	double sum,trace;
    
    IDX = mxGetPr(prhs[0]);
    B = mxGetPr(prhs[1]);
    X = mxGetPr(prhs[2]);
    beta = *(mxGetPr(prhs[3]));
     
    knn = mxGetN(prhs[0]);
    N = mxGetM(prhs[0]);
    K = mxGetM(prhs[1]);
    D = mxGetN(prhs[1]);
            
    plhs[0] = mxCreateNumericMatrix(N,K,mxDOUBLE_CLASS,mxREAL);
    C = mxGetPr(plhs[0]);
    z = (double*)mxCalloc(knn*D,sizeof(double));
    c = (double*)mxCalloc(knn*knn,sizeof(double));
    cinv = (double*)mxCalloc(knn*knn,sizeof(double));
    idx = (int*)mxCalloc(knn,sizeof(int));
    w = (double*)mxCalloc(knn,sizeof(double));
    
    for (i = 0; i < N; i++){
        /*idx = IDX(i,:)*/
        for (ii = 0; ii <knn; ii++){
            idx[ii] = (int)IDX[i+ii*N];
            w[ii] = 0;
        }
       
        /*calculate z = B(idx,:) - repmat(X(i,:), knn, 1);*/
        for (ii = 0; ii < knn; ii++)
            for (j = 0; j < D; j ++)
                z[ii+j*knn] = B[idx[ii]+j*K]-X[i+j*N];
        /*calculate C = z*z';*/
        for (ii = 0; ii < knn; ii++)
            for ( iii = 0; iii < knn; iii++){
                sum = 0.0;
                for ( j = 0; j < D; j ++)
                     sum += z[ii+j*knn]*z[iii+j*knn];
                c[ii+iii*knn] = sum;
            }
        trace = 0.0;
        for (ii = 0; ii < knn; ii++)
            trace += c[ii*knn+ii];
        trace = beta*(trace+1e-16);
        for (ii = 0; ii < knn; ii++)
            c[ii*knn+ii] += trace;
        inv(c,cinv,knn);
        for (ii = 0; ii < knn; ii++)
            for (jj = 0; jj < knn; jj++)
                w[ii] += cinv[ii+jj*knn];
        sum = 0;
        for (ii = 0; ii < knn; ii++)
            sum += w[ii];
        
        for (ii = 0; ii < knn; ii++)
            w[ii] /= sum;
        for (ii = 0; ii < knn; ii++)
            C[i+idx[ii]*N] = w[ii];
           
        
        
        
        
        
    }
    
     mxFree(z);
      mxFree(c);
       mxFree(cinv);
       mxFree(idx);
       mxFree(w);

  
}
