import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

public class SineExperiment {
    public static void main(String[] args) {
        // Generate 500 input vectors
        ArrayList<TrainingExample> dataset = new ArrayList<>();
        int numSamples = 500;
        Random rand = new Random();

        for (int i = 0; i < numSamples; i++) {
            double[] input = new double[4];
            for (int j = 0; j < 4; j++) {
                input[j] = rand.nextDouble() * 2 - 1; // Random between -1 and 1
            }
            double x1 = input[0];
            double x2 = input[1];
            double x3 = input[2];
            double x4 = input[3];
            double[] output = new double[1];
            output[0] = Math.sin(x1 - x2 + x3 - x4);
            dataset.add(new TrainingExample(input, output));
        }

        // Split dataset into 400 training and 100 testing samples
        ArrayList<TrainingExample> trainingData = new ArrayList<>(dataset.subList(0, 400));
        ArrayList<TrainingExample> testData = new ArrayList<>(dataset.subList(400, 500));

        // MLP configuration: 4 inputs, 5 hidden units, 1 output
        int NI = 4;
        int NH = 5;
        int NO = 1;
        double learningRate = 0.01;
        int maxEpochs = 5000;

        MLP nn = new MLP(NI, NH, NO, ActivationFunctionType.LINEAR);

        // Training loop
        double trainingError = 0;
        Random randShuffle = new Random();
        int batchSize = 5; // Update weights after every 5 examples
        int batchCounter = 0;
        ArrayList<Double> epochErrors = new ArrayList<>(); // To log errors for specific epochs

        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            trainingError = 0; // Reset error at the beginning of each epoch

            // Shuffle training data (optional)
            java.util.Collections.shuffle(trainingData, randShuffle);

            // Iterate through training data
            for (TrainingExample example : trainingData) {
                nn.forward(example.input); // Forward pass
                trainingError += nn.backward(example.input, example.output); // Accumulate updates
                batchCounter++;

                // Update weights after every batchSize examples
                if (batchCounter % batchSize == 0) {
                    nn.updateWeights(learningRate);
                }
            }

            // Apply remaining updates
            if (batchCounter % batchSize != 0) {
                nn.updateWeights(learningRate);
            }

            batchCounter = 0; // Reset for next epoch

            // Log error for specific epochs
            if (epoch % 500 == 0 || epoch == maxEpochs - 1) {
                epochErrors.add(trainingError);
            }
        }

        // Evaluate on test set
        double testError = 0;
        ArrayList<String> testPredictions = new ArrayList<>();
        for (TrainingExample example : testData) {
            nn.forward(example.input);
            double diff = nn.O[0] - example.output[0];
            testError += diff * diff;

            // Log some example calculations for predictions
            testPredictions.add(String.format(
                "Input: %s, Expected: %.4f, Predicted: %.4f, Squared Error: %.4f",
                java.util.Arrays.toString(example.input),
                example.output[0],
                nn.O[0],
                diff * diff
            ));
        }
        testError /= 2.0;

        // Write results to a file
        try (BufferedWriter writer = new BufferedWriter(new FileWriter("SineExperimentResults.txt"))) {
            writer.write("Sine Experiment Results\n");
            writer.write("========================\n");
            writer.write("Configuration:\n");
            writer.write("Number of Inputs: " + NI + "\n");
            writer.write("Number of Hidden Units: " + NH + "\n");
            writer.write("Number of Outputs: " + NO + "\n");
            writer.write("Learning Rate: " + learningRate + "\n");
            writer.write("Max Epochs: " + maxEpochs + "\n");
            writer.write("Activation Function: LINEAR\n\n");

            writer.write("Training Error Over Selected Epochs:\n");
            writer.write("Epoch\tError\n");
            for (int i = 0; i < epochErrors.size(); i++) {
                int epochLogged = i * 500; // Log every 500 epochs
                writer.write(epochLogged + "\t" + epochErrors.get(i) + "\n");
            }
            writer.write("\nFinal Training Error: " + trainingError + "\n");
            writer.write("Final Test Error: " + testError + "\n\n");

            writer.write("Sample Calculations from Test Set:\n");
            for (int i = 0; i < Math.min(5, testPredictions.size()); i++) {
                writer.write(testPredictions.get(i) + "\n");
            }

            System.out.println("Results saved to SineExperimentResults.txt");
        } catch (IOException e) {
            System.err.println("Error writing to file: " + e.getMessage());
        }
    }
}
