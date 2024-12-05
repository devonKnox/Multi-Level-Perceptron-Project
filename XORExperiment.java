import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class XORExperiment {
    public static void main(String[] args) {
        // Training data for XOR function
        ArrayList<TrainingExample> trainingData = new ArrayList<>();
        trainingData.add(new TrainingExample(new double[]{0, 0}, new double[]{0}));
        trainingData.add(new TrainingExample(new double[]{0, 1}, new double[]{1}));
        trainingData.add(new TrainingExample(new double[]{1, 0}, new double[]{1}));
        trainingData.add(new TrainingExample(new double[]{1, 1}, new double[]{0}));

        // MLP configuration: 2 inputs, 4 hidden units, 1 output
        int NI = 2;
        int NH = 4;
        int NO = 1;
        double learningRate = 1;
        int maxEpochs = 2000;
        ActivationFunctionType activationFunction = ActivationFunctionType.LINEAR;

        MLP nn = new MLP(NI, NH, NO, activationFunction);

        // Log training error for various epochs
        ArrayList<Double> errorLog = new ArrayList<>();
        int loggingInterval = 50;

        // Training loop
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            double error = 0;

            // Iterate through training data
            for (TrainingExample example : trainingData) {
                nn.forward(example.input); // Forward pass
                error += nn.backward(example.input, example.output); // Backward pass
            }

            nn.updateWeights(learningRate); // Update weights

            // Log error at specified intervals
            if (epoch % loggingInterval == 0) {
                errorLog.add(error);
            }
        }

        // Calculate final overall error
        double finalError = 0;
        for (TrainingExample example : trainingData) {
            nn.forward(example.input);
            double diff = nn.O[0] - example.output[0];
            finalError += diff * diff;
        }
        finalError = Math.sqrt(finalError / trainingData.size()) * 100;


        // Write results to file
        try (BufferedWriter writer = new BufferedWriter(new FileWriter("XORExperimentResults.txt"))) {
            writer.write("XOR Experiment Results\n");
            writer.write("=======================\n");
            writer.write("Configuration:\n");
            writer.write("Number of Inputs: " + NI + "\n");
            writer.write("Number of Hidden Units: " + NH + "\n");
            writer.write("Number of Outputs: " + NO + "\n");
            writer.write("Learning Rate: " + learningRate + "\n");
            writer.write("Max Epochs: " + maxEpochs + "\n");
            writer.write("Activation Function: " + activationFunction + "\n\n");

            writer.write("Training Error Over Epochs (Logged Every " + loggingInterval + " Epochs):\n");
            writer.write("Epoch\tError\n");
            for (int i = 0; i < errorLog.size(); i++) {
                writer.write((i * loggingInterval) + "\t" + errorLog.get(i) + "\n");
            }

            writer.write("\nResults:\n");
            for (TrainingExample example : trainingData) {
                nn.forward(example.input);
                String inputString = java.util.Arrays.toString(example.input);
                writer.write(String.format(
                    "Input: %s, Expected Output: %.1f, Predicted Output: %.4f\n",
                    inputString,
                    example.output[0],
                    nn.O[0]
                ));
            }

            writer.write("\nFinal Root Mean Squared Error: " + String.format("%.2f%%", finalError));
            System.out.println("Results saved to XORExperimentResults.txt");
        } catch (IOException e) {
            System.err.println("Error writing to file: " + e.getMessage());
        }
    }
}