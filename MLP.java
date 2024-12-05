import java.util.Random;

public class MLP {
    int NI, NH, NO; // Number of inputs, hidden units, and outputs
    double[][] W1, W2; // Weights in lower and upper layers
    double[][] dW1, dW2; // weight changes applied to W1 and W2
    double[] Z1, Z2, H, O; // Activations for lower layer, Activations for upper layer, values of hidden neurons (for dW2), output array
    ActivationFunctionType outputActivation;

    public MLP(int numInputs, int numHidden, int numOutputs, ActivationFunctionType outputActivation) {
        NI = numInputs;
        NH = numHidden;
        NO = numOutputs;
        this.outputActivation = outputActivation;

        W1 = new double[NI][NH];
        W2 = new double[NH][NO];
        dW1 = new double[NI][NH];
        dW2 = new double[NH][NO];
        Z1 = new double[NH];
        Z2 = new double[NO];
        H = new double[NH];
        O = new double[NO];

        randomise();
    }

    // Initialises W1 and W2 to small random values and dW1 and dW2 to zeros
    public void randomise() {
        Random rand = new Random();
        double range = 1.0;
        for (int i = 0; i < NI; i++) {
            for (int j = 0; j < NH; j++) {
                W1[i][j] = rand.nextDouble() * 2 * range - range;
                dW1[i][j] = 0.0;
            }
        }
        for (int j = 0; j < NH; j++) {
            for (int k = 0; k < NO; k++) {
                W2[j][k] = rand.nextDouble() * 2 * range - range;
                dW2[j][k] = 0.0;
            }
        }
    }

    // Forward propagation; input[] processed to produce output in O[]
    public void forward(double[] input) {
        // Hidden layer
        for (int j = 0; j < NH; j++) {
            Z1[j] = 0.0;
            for (int i = 0; i < NI; i++) {
                Z1[j] += input[i] * W1[i][j];
            }
            H[j] = sigmoid(Z1[j]);
        }

        // Output layer
        for (int k = 0; k < NO; k++) {
            Z2[k] = 0.0;
            for (int j = 0; j < NH; j++) {
                Z2[k] += H[j] * W2[j][k];
            }
        }

        // Apply respective activation function
        if (outputActivation == ActivationFunctionType.SIGMOID) {
            for (int k = 0; k < NO; k++) {
                O[k] = sigmoid(Z2[k]);
            }
        } else if (outputActivation == ActivationFunctionType.LINEAR) {
            for (int k = 0; k < NO; k++) {
                O[k] = Z2[k];
            }
        } else if (outputActivation == ActivationFunctionType.SOFTMAX) {
            O = softmax(Z2);
        }
    }

    // Backward propagation
    public double backward(double[] input, double[] target) {
        double error = 0.0;
        double[] deltaO = new double[NO];

        if (outputActivation == ActivationFunctionType.SIGMOID || outputActivation == ActivationFunctionType.LINEAR) {
            for (int k = 0; k < NO; k++) {
                double diff = O[k] - target[k];
                if (outputActivation == ActivationFunctionType.SIGMOID) {
                    deltaO[k] = diff * sigmoidDerivative(O[k]);
                    error += 0.5 * diff * diff; // Mean Squared Error
                } else if (outputActivation == ActivationFunctionType.LINEAR) {
                    deltaO[k] = diff;
                    error += 0.5 * diff * diff;
                }
            }
        } else if (outputActivation == ActivationFunctionType.SOFTMAX) {
            // Cross-Entropy Loss
            for (int k = 0; k < NO; k++) {
                deltaO[k] = O[k] - target[k]; // Cross-Entropy derivative
                error -= target[k] * Math.log(O[k]);
            }
        }

        // Hidden layer delta
        double[] deltaH = new double[NH];
        for (int j = 0; j < NH; j++) {
            deltaH[j] = 0.0;
            for (int k = 0; k < NO; k++) {
                deltaH[j] += deltaO[k] * W2[j][k];
            }
            deltaH[j] *= sigmoidDerivative(H[j]);
        }

        // Accumulate W2 weight updates
        for (int j = 0; j < NH; j++) {
            for (int k = 0; k < NO; k++) {
                dW2[j][k] += deltaO[k] * H[j];
            }
        }

        // W1
        for (int i = 0; i < NI; i++) {
            for (int j = 0; j < NH; j++) {
                dW1[i][j] += deltaH[j] * input[i];
            }
        }

        return error;
    }

    // Update weights with accumulated updates
    public void updateWeights(double learningRate) {
        // Update W2
        for (int j = 0; j < NH; j++) {
            for (int k = 0; k < NO; k++) {
                W2[j][k] -= learningRate * dW2[j][k];
                dW2[j][k] = 0.0; // Reset accumulated update
            }
        }

        // Update W1
        for (int i = 0; i < NI; i++) {
            for (int j = 0; j < NH; j++) {
                W1[i][j] -= learningRate * dW1[i][j];
                dW1[i][j] = 0.0; // Reset
            }
        }
    }

    // Sigmoid activation function
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    // Sigmoid derivative
    private double sigmoidDerivative(double output) {
        return output * (1.0 - output);
    }

    // Softmax activation function
    private double[] softmax(double[] z) {
        double max = Double.NEGATIVE_INFINITY;
        for (double val : z) {
            if (val > max) {
                max = val;
            }
        }
        double sum = 0.0;
        double[] expVals = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            expVals[i] = Math.exp(z[i] - max);
            sum += expVals[i];
        }
        for (int i = 0; i < z.length; i++) {
            expVals[i] /= sum;
        }
        return expVals;
    }
}
