/**
 * 
 * @author Peter Lawson
 * @version Final Exam
 */
public class complex {

	private double real = 0;
	private double imag = 0;

	/**
	 * complex class constructor with arguments.
	 * 
	 * @param r
	 * @param i
	 */
	public complex(double r, double i) {
		real = r;
		imag = i;

	}

	/**
	 * complex class constructor without arguments.
	 */
	public complex() {
		this.real = 0;
		this.imag = 0;
	}

	/**
	 * Setter for real value in complex number.
	 * 
	 * @param real
	 */
	public void setReal(double real) {
		this.real = real;
	}

	/**
	 * Getter for real value in complex number.
	 * 
	 * @return
	 */
	public double getReal() {
		return this.real;

	}

	/**
	 * Setter for imaginary number in complex number.
	 * 
	 * @param imag
	 */
	public void setImag(double imag) {
		this.imag = imag;

	}

	/**
	 * Getter for imaginary number in complex number.
	 * 
	 * @return
	 */
	public double getImag() {
		return this.imag;
	}

	/**
	 * Prints to string a complex number.
	 */
	public String toString() {
		if (this.real == 0) {
			if (this.imag == 0) {
				return "0";
			} else {
				return (this.imag + "i");
			}
		} else {
			if (this.imag == 0) {
				return String.valueOf(this.real);
			} else if (this.imag < 0) {
				return (this.real + " " + this.imag + "i");
			} else {
				return (this.real + " +" + this.imag + "i");
			}
		}

	}

	/**
	 * Returns the conjugate of a complex number.
	 * 
	 * @return
	 */
	public complex conjugate() {
		return new complex(this.real, this.imag * (-1));
	}

	/**
	 * Adds two complex numbers.
	 * 
	 * @param input
	 * @return
	 */
	public complex add(complex input) {
		complex sum = new complex();
		sum.setReal(this.real + input.getReal());
		sum.setImag(this.imag + input.getImag());
		return sum;

	}

	/**
	 * Computes the modulus of a complex number.
	 * 
	 * @param input
	 * @return
	 */
	public double modulus(complex input) {
		if (input.getReal() != 0 || input.getImag() != 0) {
			return Math.sqrt(input.getReal() * input.getReal()
					+ input.getImag() * input.getImag());
		} else {
			return 0;
		}
	}

	/**
	 * Subtracts two complex numbers.
	 * 
	 * @param input
	 * @return
	 */
	public complex subtract(complex input) {
		complex difference = new complex();
		difference.setReal(this.real - input.getReal());
		difference.setImag(this.imag - input.getImag());
		return difference;
	}

	/**
	 * Multiplies two complex numbers.
	 * 
	 * @param input
	 * @return
	 */
	public complex multiply(complex input) {
		complex product = new complex();
		product.setReal(this.real * input.getReal() - this.imag
				* input.getImag());
		product.setImag(this.real * input.getImag() + this.imag
				* input.getReal());
		return product;
	}

	/**
	 * Divides two complex numbers.
	 * 
	 * @param input
	 * @return
	 */
	public complex divide(complex input) {
		double placeholder = Math.pow(modulus(input), 2);
		complex quotient = new complex();
		quotient.setReal((this.real * input.getReal() + this.imag
				* input.getImag())
				/ placeholder);
		quotient.setImag((this.imag * input.getReal() - this.real
				* input.getImag())
				/ placeholder);
		return quotient;
	}

}
