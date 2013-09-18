import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Scanner;

public class Fourier {
	static int linecount = 0;
	static ArrayList<Double> data = new ArrayList<Double>();
	static ArrayList<complex> complexdata = new ArrayList<complex>();

	/**
	 * @param args
	 */
	public static void main(String arg[]) {
		try {
			// counts the lines in input.txt
			Scanner lineScan = new Scanner(new File("input.txt"));

			while (lineScan.hasNextLine()) {
				lineScan.nextLine();
				linecount++;
			}

			lineScan.close();

			// reads in input.txt into a series of matrix vectors divided into
			// two classes.
			Scanner scan = new Scanner(new File("input.txt"));
			for (int e = 0; e < linecount; e++) {
				double val = scan.nextDouble();
				data.add(val);
			}

			scan.close();
			for (int l = 0; l < linecount; l++) {
				complexdata.add(new complex(data.get(l), 0));
			
			}
			// end try
		} catch (NumberFormatException e) {
			System.out.println("The input file contains a non-integer.");
		} catch (Exception e) {
			System.err.println(e.getMessage());

		}
		double theta = 0;
		double z5=0;
		double z10=0;
		double z100=0;

		int N = linecount;
		theta = (-2 * Math.PI * 1) / N;
		int r = N / 2;
		complex t = new complex(0, 0);
		for (int i = 1; i < N; i *= 2) {
			complex w = new complex(Math.cos(i * theta), Math.sin(i * theta));
			for (int k = 0; k < N; k += 2 * r) {
				complex u = new complex(1, 0);
				for (int m = 0; m < r; m++) {
					t = complexdata.get(k+m).subtract(complexdata.get(k+m+r));
					complexdata.set(k+m, complexdata.get(k+m).add(complexdata.get(k+m+r)));
					complexdata.set(k+m+r, t.multiply(u));
					u=w.multiply(u);

				}
			}
			r=r/2;
		}
		
		for (int i = 0; i < N; i++) {
			r=i;
			int k=0;
		for (int m=1;m<N;m*=2)
			{
				k= 2*k +(r % 2);
				r =r/2;
			}
			if (k>i)
			{
				t=complexdata.get(i);
				complexdata.set(i, complexdata.get(k));
				complexdata.set(k, t);
			}
			
		}
		for(double x=0.000976562;x<1;x+=0.000976562)
		{
		for(int k =1; k<=5;k++)
		{
			
			 z5 += Math.sin(2*Math.PI*(2*k-1)*x)/((2*k)-1);
		}
			try {
	 		    PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("output5a.txt", true)));
	 		   out.println(z5);
	 		 
	 		 
	 		    out.close();
	 		} catch (IOException e) {
	 		    
	 		}
			
			z5=0;
		
		}
		for(double x=0.000976562;x<1;x+=0.000976562)
		{
		for(int k =1; k<=10;k++)
		{
			
			 z10 += Math.sin(2*Math.PI*(2*k-1)*x)/((2*k)-1);
		}
			try {
	 		    PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("output10a.txt", true)));
	 		   out.println(z10);
	 		 
	 		 
	 		    out.close();
	 		} catch (IOException e) {
	 		    
	 		}
			
			z10=0;
		
		}
		for(double x=0.000976562;x<1;x+=0.000976562)
		{
		for(int k =1; k<=100;k++)
		{
			
			 z100 += Math.sin(2*Math.PI*(2*k-1)*x)/((2*k)-1);
		}
			try {
	 		    PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("output100a.txt", true)));
	 		   out.println(z100);
	 		 
	 		 
	 		    out.close();
	 		} catch (IOException e) {
	 		    
	 		}
			
			z100=0;
		
		}
		
		
		
		for(double x=0.000976562;x<1;x+=0.000976562)
		{
		for(int k =1; k<=5;k++)
		{
			
			 z5 += Math.sin(2*Math.PI*(2*k)*x)/((2*k));
		}
			try {
	 		    PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("output5b.txt", true)));
	 		   out.println(z5);
	 		 
	 		 
	 		    out.close();
	 		} catch (IOException e) {
	 		    
	 		}
			
			z5=0;
		
		}
		for(double x=0.000976562;x<1;x+=0.000976562)
		{
		for(int k =1; k<=10;k++)
		{
			
			 z10 += Math.sin(2*Math.PI*(2*k)*x)/((2*k));
		}
			try {
	 		    PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("output10b.txt", true)));
	 		   out.println(z10);
	 		 
	 		 
	 		    out.close();
	 		} catch (IOException e) {
	 		    
	 		}
			
			z10=0;
		
		}
		for(double x=0.000976562;x<1;x+=0.000976562)
		{
		for(int k =1; k<=100;k++)
		{
			
			 z100 += Math.sin(2*Math.PI*(2*k)*x)/((2*k));
		}
			try {
	 		    PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("output100b.txt", true)));
	 		   out.println(z100);
	 		 
	 		 
	 		    out.close();
	 		} catch (IOException e) {
	 		    
	 		}
			
			z100=0;
		
		}
		
		
		System.out.println("Linecount is " + linecount);
		
		for (int x=0;x<N;x++)
		{
			System.out.println(complexdata.get(x));
		}
		/*
		 * complex a = new complex(3, 4); complex b = new complex(2, -6);
		 * System.out.println(a); System.out.println(b);
		 * System.out.println(" add " +a.add(b));
		 * System.out.println(" subtract " + a.subtract(b));
		 * System.out.println(" multiply " + a.multiply(b));
		 * System.out.println(" divide " + a.divide(b));
		 * System.out.println(a.divide(b).multiply(b));
		 */

	}

}
