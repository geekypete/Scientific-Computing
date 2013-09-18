import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

/**
 * This program performs matrix operations to facilitate the solving of a system
 * of equations as well as predict what functions a given vector belongs to.
 * 
 * @author Peter Lawson
 * @version Exam 2
 */
final public class matrix {
	final int row;
	final int col;
	final double[][] array;

	/**
	 * Matrix constructor.
	 * 
	 * @param row
	 *            number of rows in matrix.
	 * @param col
	 *            number of cols in matrix.
	 * 
	 */
	public matrix(int row, int col) {
		this.row = row;
		this.col = col;
		array = new double[row][col];
	}

	/**
	 * Allows a matrix to be copied when called in method.
	 * 
	 * @param matrix
	 *            A.
	 * 
	 * 
	 * */
	private matrix(matrix A) {
		this(A.array);
	}

	/**
	 * Generates a matrix from a 2-D array.
	 * 
	 * @param  array.
	 * 
	 * */
	public matrix(double[][] array) {
		row = array.length;
		col = array[0].length;
		this.array = new double[row][col];
		for (int i = 0; i < row; i++)
			for (int j = 0; j < col; j++)
				this.array[i][j] = array[i][j];
	}

	/**
	 * Generates exception if the wrong matrix sizes are utilized when a matrix
	 * operation is called.
	 * */
	public class InvalidMatrixSize extends Exception {

		public InvalidMatrixSize(String message) {
			super(message);
		}

	}

	/**
	 * Generates a matrix identity of NxN size.
	 * 
	 * @return matrix identity : a matrix identity of size N.
	 * @param  size : the N size of the matrix identity to be generated;
	 * 
	 */
	public static matrix identitygenerator(int size) {
		matrix identity = new matrix(size, size);
		for (int j = 0; j < size; j++)
			identity.array[j][j] = 1;
		return identity;
	}

	/**
	 * Generates a matrix identity of NxN size.
	 * 
	 * @return matrix A : a transpose of the matrix calling the method
	 *         transpose().
	 * 
	 */
	public matrix transpose() {
		matrix A = new matrix(col, row);
		for (int i = 0; i < row; i++)
			for (int j = 0; j < col; j++)
				A.array[j][i] = this.array[i][j];
		return A;
	}

	/**
	 * The guassJordan algorithm: takes a Nx(N+1) augmented matrix and returns
	 * an NxN identity and a Nx1 set of solutions if such solutions do exist.
	 * Otherwise it outputs an error and terminates.
	 * 
	 * @param B :An augmented matrix of size Nx(N+1);
	 * @return matrix A : an identity of NxN and the Nx1 column consisting of
	 *         solutions to a set of equations if they exist.
	 * 
	 */
	private matrix gaussJordan(matrix B) {
		matrix A = new matrix(B);
		int E = 1;
		for (int j = 0; j < A.row; j++) {

			int pivot = j;
			for (int i = j + 1; i < A.row; i++) {
				if (Math.abs(A.array[i][j]) > Math.abs(A.array[pivot][j])) {
					pivot = i;
					double[] temp = A.array[j];
					A.array[j] = A.array[pivot];
					A.array[pivot] = temp;
					if (pivot == 0) {
						E = 0;
					}
					if (E == 0) {
						System.err.println("No single solution exists!");
						break;
					}
				}
			}

			for (int i = 0; i < A.row; i++) {
				double div = A.array[i][j] / A.array[j][j];
				for (int jj = 0; jj <= A.row; jj++) {
					if (i != j && jj != j) {
						A.array[i][jj] -= div * A.array[j][jj];
					}
				}
			}

			for (int i = 0; i < A.row; i++) {
				if (i != j)
					A.array[i][j] = 0.0;
			}
			for (int jj = 0; jj <= A.row; jj++) {
				if (jj != j) {
					A.array[j][jj] /= A.array[j][j];
				}
			}
			A.array[j][j] = 1.0;
		}

		return A;
	}

	/**
	 * Takes the inverse of a NxN matrix.
	 * 
	 * @param matrix
	 *            z: a matrix of NxN size for which the inverse of that matrix
	 *            is to be calculated.
	 * @return matrix inverse : Returns an NxN matrix representing the inverse
	 *         of the matrix parameter.
	 */
	public static matrix inverse(matrix Z) {
		matrix Y = new matrix(Z);
		matrix A = Y.transpose();
		int E = 1;
		matrix identity = new matrix(identitygenerator(A.row));
		// This takes an identity and puts it on the the matrix inputed so that
		// an inverse can be computed.
		double m[][] = Arrays.copyOf(A.array, identity.array.length
				+ A.array.length);
		System.arraycopy(identity.array, 0, m, A.array.length,
				identity.array.length);
		matrix R = new matrix(m);
		matrix B = R.transpose();

		for (int j = 0; j < B.row; j++) {

			int pivot = j;
			for (int i = j + 1; i < B.row; i++) {
				// this takes the computed pivot and swaps it to put the max row
				// first.
				if (Math.abs(B.array[i][j]) > Math.abs(B.array[pivot][j])) {
					pivot = i;
					double[] temp = A.array[j];
					A.array[j] = A.array[pivot];
					A.array[pivot] = temp;
					if (pivot == 0) {
						E = 0;
					}
					if (E == 0) {
						System.err.println("This inverse does not exist!");
						break;
					}

				}
			}

			// Performs a modified Gauss-Jordan to compute inverse
			for (int i = 0; i < B.row; i++) {
				double div = B.array[i][j] / B.array[j][j];
				for (int jj = 0; jj <= B.col - 1; jj++) {
					if (i != j && jj != j) {
						B.array[i][jj] -= div * B.array[j][jj];
					}
				}
			}

			for (int i = 0; i < B.row; i++) {
				if (i != j)
					B.array[i][j] = 0.0;
			}
			for (int jj = 0; jj <= B.col - 1; jj++) {
				if (jj != j) {
					B.array[j][jj] /= B.array[j][j];
				}
			}
			B.array[j][j] = 1.0;
		}
		double inv[][] = new double[B.row][B.row];
		for (int i = 0; i < B.row; i++) {
			for (int j = 0; j < B.row; j++) {
				inv[i][j] = B.array[i][j + B.row];
			}
		}
		matrix inverse = new matrix(inv);
		return inverse;

	}///////////////////////////////////////////////////////////////////////////////////////
	
	public static double secant(double a,double b, double c, double d, double rangeA, double rangeB)
	{

		double epsilon =.0001;
		int m =100;
		double x=0;
		double x0=rangeA;
		double x1=rangeB;
		int k=1;
		double f0=a*x0*x0*x0+b*x0*x0+c*x0+d; 
		double f1=a*x1*x1*x1+b*x1*x1+c*x1+d; 
		
		do
		{
		double x2=x1-(((x1-x0)/(f1-f0))*f1);
		x0=x1;
		f0=f1;
		x1=x2;
		f1=a*x2*x2*x2+b*x2*x2+c*x2+d; 
		k=k+1;
		x=x2;
		}
		while(Math.abs(f1)>=epsilon && k<=m);
		
		
		
		return x;
		
	}
	
	
	
	/**
	 * Takes the mean of a set of vectors.
	 * 
	 * @param matrix A: an arraylist consisting of all the vectors for which the
	 *            means is to be calculated.
	 * @return matrix avg : A vector representing the mean of all the inputted
	 *         vectors.
	 */
	public static matrix average(ArrayList A) {
		double[][] sum = new double[2][1];
		sum[0][0] = 0;
		sum[1][0] = 0;
		matrix sumvector = new matrix(sum);

		for (int z = 0; z < A.size(); z++) {
			matrix c = new matrix((double[][]) A.get(z));
			sumvector = sumvector.add(c);

		}

		matrix avg = scalar(sumvector, (1 / (double) A.size()));

		return avg;

	}

	/**
	 * Finds the covariance for a set of vectors.
	 * 
	 * @param matrix
	 *            A: an arraylist consisting of all the vectors for which the
	 *            covariance is to be calculated.
	 * @return matrix covar: A matrix representing the covariance for the set of
	 *         inputed vectors.
	 */
	public static matrix covariance(ArrayList A) {
		double[][] sum = new double[2][2];
		// takes each vector, subtracts the avg for that class, multiplies it by
		// its transpose, and then sums each one, than multiplies by a scalar of
		// (1/total vectors)
		matrix sumvector = new matrix(sum);
		for (int z = 0; z < A.size(); z++) {
			matrix avg = new matrix(average(A));
			matrix a = new matrix((double[][]) A.get(z));
			matrix b = new matrix(a.subtract(avg));
			matrix c = b.transpose();
			matrix d = b.times(c);
			sumvector = sumvector.add(d);

		}

		matrix covar = scalar(sumvector, (1 / (double) A.size()));

		return covar;

	}

	/**
	 * Multiplies a matrix by a scalar value.
	 * 
	 * @return matrix C : the resulting matrix after multiplication by the
	 *         scalar.
	 * @param matrix
	 *            A: the matrix to be multiplied by scalar.
	 * @param double s; the scalar to multiply the matrix by.
	 * 
	 */
	public static matrix scalar(matrix A, double s) {

		matrix C = new matrix(A.row, A.col);
		// multiplies each value in the array by a scalar value and returns the
		// modified matrix.
		for (int i = 0; i < C.row; i++)
			for (int j = 0; j < C.col; j++)
				C.array[i][j] += (A.array[i][j] * s);
		return C;
	}

	/**
	 * Adds two matrices.
	 * 
	 * @return matrix c : the sum of the matrix entered and the calling matrix.
	 * @param matrix B: the matrix to add.
	 * @throws InvalidMatrixSize
	 *             if the two matrices are not the same size.
	 */
	public matrix add(matrix B) {
		matrix A = this;
		if (A.row != B.row || A.col != B.col)
			try {
				throw new InvalidMatrixSize(
						"These matrices are not the same size!");
			} catch (InvalidMatrixSize e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		// each value from one matrix to another and returns the modified
		// matrix.
		matrix C = new matrix(row, col);
		for (int i = 0; i < row; i++)
			for (int j = 0; j < col; j++)
				C.array[i][j] = A.array[i][j] + B.array[i][j];
		return C;
	}

	/**
	 * Subtracts the inputed matrix from the calling matrix.
	 * 
	 * @return matrix c : the resultant matrix after subtracting the inputed
	 *         matrix from the calling matrix.
	 * @param matrix
	 *            B the matrix to subtract.
	 * 
	 * @throws InvalidMatrixSize
	 *             if the two matrices are not the same size.
	 */
	public matrix subtract(matrix B) {
		matrix A = this;
		if (B.row != A.row || B.col != A.col)
			try {
				throw new InvalidMatrixSize(
						"These matrices are not the same size!");
			} catch (InvalidMatrixSize e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		matrix C = new matrix(row, col);
		for (int i = 0; i < row; i++)
			for (int j = 0; j < col; j++)
				C.array[i][j] = A.array[i][j] - B.array[i][j];
		return C;
	}

	/**
	 * Multiplies the inputed matrix by the calling matrix.
	 * 
	 * @return matrix C : the resultant matrix after multiplying the inputed
	 *         matrix by the calling matrix.
	 * @param matrix
	 *            B : the matrix to multiply.
	 * 
	 * @throws InvalidMatrixSize
	 *             if the number of columns of the calling matrix is not equal
	 *             to the number of the rows of the inputed matrix.
	 */
	public matrix times(matrix B) {
		matrix A = this;
		if (A.col != B.row)
			try {
				throw new InvalidMatrixSize(
						"The number of columns of the 1st matrix must equal the number of rows of the 2nd matrix!");
			} catch (InvalidMatrixSize e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		// multiples each value in one matrix by the same located value in
		// another matrix and then returns the modified matrix.
		matrix C = new matrix(A.row, B.col);
		for (int i = 0; i < C.row; i++)
			for (int j = 0; j < C.col; j++)
				for (int k = 0; k < A.col; k++)
					C.array[i][j] += (A.array[i][k] * B.array[k][j]);
		return C;
	}

	/**
	 * Outputs the calling matrix by iterating through each row/col and printing
	 * the value to screen.
	 * 
	 */
	public void output() {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++)
				System.out.print(" " + array[i][j]);
			System.out.println();
		}
	}
	
	/**
	 * Outputs the trace of a matrix.
	 * @return double trace
	 * @param matrix a
	 */
	public static double trace(matrix a)
	{
		double trace=0;
	for(int i = 0; i < a.row; i++)
	{
			trace += a.array[i][i];
	
	}
	return trace;
	}
	
	/**
	 * Performs Leverrier's method to identify coefficients in polynomial function.
	 * @param matrix A
	 * @return String equation
	 */	
	public static void leverrier(matrix A) {
		matrix B= new matrix(A);
		double a = -trace(B);
		
		System.out.print("x^"+ A.row + " " +a+ "X^"+ (A.row-1)+" ");
		
		for(int k=B.row-1;k>=1;k--)
		{
			 
			matrix I= identitygenerator(B.row);
			
			for(int i =0; i<B.row;i++)
			{
				I.array[i][i]*=a;

			}
		
			matrix BplusaI = B.add(I);
		     B=A.times(BplusaI);
			 a = -(trace(B)/(A.row-k+1));
			 System.out.print(" + " + a+ "X^"+(k-1)+ " ");
			
			
		}
		System.out.println();
		
		
	}
	
	

	/**
	 * Performs power method to estimate largest eigenvalue for the inputed matrix
	 * @param matrix A : the matrix for which the eigenvalue is being estimated.
	 * @return double eigenvalue
	 */	
	public static double powerMethod(matrix B) {
		
		matrix A= new matrix(B);
		double epsilon =1.0E-15;
		int m=100;
		int k=0;
		double[][] test =new double[A.row][1];
		double norm=0;
		double norm2=0;
		double temp=0;
		double temp2=0;
		double u =0;
		//Populate arbitrary vector
		for (int i=0;i<A.row;i++)
		
			{
			
				test[i][0]=2;
			}
		//declare y , an arbitrary vector
		matrix y= new matrix(test);
		
		//now compute x=A.times(y)
	matrix x= A.times(y);
	do
	{
		//Square and sum all values in x vector to compute norm
		for (int i=0;i<A.row;i++)
			
		{
		
			temp += Math.pow(test[i][0], 2);
			
		}
		//Square root x vector to get norm value!
			norm=Math.sqrt(temp);
		//Normalize x!
			for (int i=0;i<A.row;i++)
				
			{
			
				x.array[i][0]/=norm;
				
				
			}	
			// y=normalized x!
			 y = x; 
		

			x = A.times(y);
			//find y transpose
			matrix yt=y.transpose();
			//find numerator of eigenvalue calc
			matrix placehold = yt.times(x);
			//find denominator of eigenvalue calc
			matrix placehold2= yt.times(y);
			//calculate eigenvalue estimate
			u = ((placehold.array[0][0])/(placehold2.array[0][0]));
			//calculate u*y
for (int i=0;i<A.row;i++)
				
			{
			
				 y.array[i][0]*=u;
				
				
			}	




//now subtract x from u*y
                matrix r = y.subtract(x);
//compute r norm
                for (int i=0;i<A.row;i++)
        			
        		{
        		
        			temp2 += Math.pow(r.array[i][0], 2);
        			
        		}
        		
        			norm2=Math.sqrt(temp2);
		
	       k=k+1;
	     
	      
	}
	
	
	while((norm2>epsilon) && (k<m));
      

		return u;
		
	}
	

	/**
	 * Performs quadratic equation
	 * @param double a : A value quadratic eq
	 * @param double b : B value quadratic eq
	 * @param double c : C value quadratic eq
	 */	
	public static double[] quadratic(double a, double b, double c) {
		double disc = Math.sqrt((b*b)-(4*a*c));
		if (disc < 0){
			System.out.println("There are no real roots!");
			return null; 
			
		}
		else{
			double[] roots=new double[2];
			roots[0] = (-b+disc)/(2*a);
			roots[1] = (-b-disc)/(2*a);
			return roots;
		}
		
	}

	/**
	 * Finds the determinant for an inputed matrix.
	 * 
	 * @param matrix
	 *            B: the matrix for which the determinant is to be calculated
	 * @return double determinant: the calculated determinant to be returned.
	 * 
	 */
	public static double determinant(matrix B) {
		matrix A = new matrix(B);
		double determinant = 0;
		double holder = 1;
		int r = 0;
		// determinant base case 1 if matrix is 1 x 1;
		if (A.row == 1) {
			determinant = A.array[0][0];
			return determinant;
		}
		// determinant base case 2 if matrix is 2x2;
		if (A.row == 2) {
			determinant = A.array[0][0] * A.array[1][1] - A.array[0][1]
					* A.array[1][0];
			return determinant;

			// if matrix is bigger than 2x2 use a modified Gaussian elimination
			// method to compute determinant.
		} else {
			int E = 1;
			for (int j = 0; j < A.row; j++) {

				int pivot = j;
				for (int i = j + 1; i < A.row; i++) {
					if (Math.abs(A.array[i][j]) > Math.abs(A.array[pivot][j])) {
						// pivot swap
						pivot = i;
						double[] temp = A.array[j];
						A.array[j] = A.array[pivot];
						A.array[pivot] = temp;
						r++;
						if (pivot == 0) {
							E = 0;
						}
						if (E == 0) {
							System.err
									.println("The determinant is 0, no single solution exists for this set of equations.");
							break;
						}
					}
				}

				for (int i = j + 1; i < A.row; i++) {
					double div = A.array[i][j] / A.array[j][j];
					for (int jj = j + 1; jj < A.row; jj++) {
						A.array[i][jj] -= A.array[j][jj] * div;
					}
					A.array[i][j] = 0.0;
				}

			}

			for (int h = 0; h < A.row; h++) {

				holder *= A.array[h][h];

			}
			// return the determinant by multiplying each value in the diagonal
			// as well as -1 to the power r.
			determinant = ((Math.pow(-1, r)) * holder);

			return determinant;

		}
	}
/////////////////////////////////////////
	
	
	
	// Method to carry out the secant search.

	
	
/////////////////////////////////////////////////	
	
	
	
	
	
	
	
	
	/**
	 * Returns a condition number for an inputed matrix.
	 * 
	 * @param matrix
	 *            B: an NxN coefficient matrix for which the condition number is
	 *            to be calculated.
	 * @return double condition: returns the calculated condition number.
	 * 
	 * 
	 */
	public static double condition(matrix B) {
		matrix A = new matrix(B);
		matrix C = new matrix(inverse(A));

		double sum = 0;
		double sum2 = 0;
		double holder = 0;
		double holder2 = 0;
		double condition = 0;

		// Sums the absolute values of each row and finds the max one and saves
		// it. Then does the same for the inverse matrix and takes these two
		// values and multiplies them.
		for (int j = 0; j < B.row; j++) {
			sum = 0;
			sum2 = 0;
			for (int i = 0; i < B.row; i++) {

				sum += Math.abs(A.array[j][i]);
				sum2 += Math.abs(C.array[j][i]);

			}
			if (sum > holder) {
				holder = sum;
			}
			if (sum2 > holder2) {
				holder2 = sum2;
			}
		}

		condition = holder * holder2;
		return condition;
	}

	/**
	 * Main method. This method reads in from 2 files, input.txt which contains
	 * the vectors for Class and Class 2 as well as from eq.txt which contains a
	 * set of equations and generates three matrices, a NxN coefficient matrix,
	 * an Nx1 constant matrix, and an Nx(N+1) augmented matrix. These inputs are
	 * used for various method calls in order to solve the questions on the
	 * exam. The output is ordered the same as on the exam, with each question
	 * listed and then the relevant output for that question.
	 * 
	 * @param args
	 * @throws INumberFormatException
	 *            if the values read in are strings not values.

	 * 
	 */
	public static void main(String[] args) {
		int linecount = 0;
		int linecount2 =0;
		ArrayList<double[][]> class1 = new ArrayList<double[][]>();
		double[][] matrix = null;
		

		try {
//counts the lines in input.txt
			Scanner lineScan = new Scanner(new File("inputmatrix.txt"));

			while (lineScan.hasNextLine()) {
				lineScan.nextLine();
				linecount++;
			}

			lineScan.close();

			//reads in input.txt into a series of matrix vectors.
			Scanner scan = new Scanner(new File("inputmatrix.txt"));
			for (int e = 0; e < linecount; e++) {
				double[][] vector = new double[2][1];
				double num = scan.nextDouble();
				vector[0][0] = num;
				double num2 = scan.nextDouble();
				vector[1][0] = num2;
				class1.add(vector);
		

			}

			scan.close();
			Scanner matrixcheck= new Scanner(new File("matrix.txt"));

			while (matrixcheck.hasNextLine()) {
				matrixcheck.nextLine();
				linecount2++;
			}

			matrixcheck.close();
			
			Scanner matrixscan = new Scanner(new File("matrix.txt"));
			
				 matrix= new double[linecount2][linecount2];
				for (int i = 0; i < linecount2; i++) {
					for (int j = 0; j < linecount2; j++) {
						matrix[i][j] = matrixscan.nextDouble();

					}
				}
			matrixscan.close();
		} // end try

		catch (NumberFormatException e) {
			System.out.println("The input file contains a non-integer.");
		} catch (Exception e) {
			System.err.println(e.getMessage());

		}
		matrix matrixin = new matrix(matrix);
	///TEST OUTPUT
		
		for(int i =-3;i<3;i++)
		{
		System.out.println(secant(-4,-7,3,4, i, i+.5));
		}
	///END TEST OUTPUT 

	///BEGIN FINAL OUTPUT
	
	/*	System.out.println("1. Eigenvectors and Eigenvalues:"+"\n");
		System.out.println("¸.·´¯`·.´¯`·.¸¸.·´¯`·.¸><(((º>");
		System.out.println("\n" + "ai: The mean vector is:" + "\n");
		average(class1).output();
		System.out.println("\n"+ "aii: The covariance matrix is:" + "\n");
		covariance(class1).output();
		System.out.println("\n"+ "b: The trace of the covariance matrix is: " +trace(covariance(class1)));
		System.out.println("\n"+ "c: The determinant of the covariance matrix is: " +determinant(covariance(class1)));
		System.out.println("\n"+ "d: The eigenvalues for the covariance matrix are: " +  -(quadratic(1.0, trace(covariance(class1)), determinant(covariance(class1)))[0])+ " and "+  -(quadratic(1.0, trace(covariance(class1)), determinant(covariance(class1)))[1]));
		System.out.println("\n"+ "e: The eigenvector for the eigenvalue "+  -(quadratic(1.0, trace(covariance(class1)), determinant(covariance(class1)))[0])+ " is [1 , 1.4019837] and the eigenvector for the eigenvalue "+  -(quadratic(1.0, trace(covariance(class1)), determinant(covariance(class1)))[1])+ " is [ 1 , -0.7132751]. These results were hand calculated and the relevant work is in the primary test document attached. \n");
		System.out.print("f: The characteristic equation for matrix A is : ");
		leverrier(matrixin);
		System.out.print("\n" + "g: The estimate for the largest eigenvalue for the matrix A is : "+powerMethod(matrixin)+ " by the power method." +"\n");	
		System.out.print("\n" + "h: The estimate for the smallest eigenvalue for the matrix A is : "+powerMethod(inverse(matrixin))+ " by the power method with the inverse of A. " + " \n");
	System.out.println("The remaining eigenvalues are 2, 3, and 4 by the rational roots theorem, the work for which is attached as a pdf.");	*/
	System.out.println();
	double[][] fulleq = new double[class1.size()][class1.size()+1];
	for (int i = 0; i < class1.size(); i++) {
			for (int j = 0; j < class1.size() + 1; j++) {
				fulleq[i][j] = 1;

			}
		}
		
		for (int i = 0; i < class1.size(); i++) {
			for (int j = 1; j < class1.size(); j++) {
				fulleq[i][j] = Math.pow(class1.get(i)[0][0], j);

			}
		}
		
		for (int j = 0; j < class1.size(); j++) {
			fulleq[j][class1.size()] = class1.get(j)[1][0];
		
		}
		
		matrix augmented = new matrix(fulleq);
		//augmented.output();
		augmented.gaussJordan(augmented).output();
	///END FINAL OUTPUT
	}

}