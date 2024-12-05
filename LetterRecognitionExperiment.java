import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;

public class LetterRecognitionExperiment {
    public static void main(String[] args) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter("LetterRecognitionExperimentResults.txt"))) {
            // Load dataset
            ArrayList<TrainingExample> dataset = loadDataset("letter-recognition.data");

            // Split into training and testing sets (80% training, 20% testing)
            int totalSamples = dataset.size();
            int trainingSamples = (int) (0.8 * totalSamples);

            ArrayList<TrainingExample> trainingData = new ArrayList<>(dataset.subList(0, trainingSamples));
            ArrayList<TrainingExample> testData = new ArrayList<>(dataset.subList(trainingSamples, totalSamples));

            // MLP configuration: 16 inputs, 40 hidden units, 26 outputs (A-Z)
            int NI = 16;
            int NH = 40;
            int NO = 26;
            double learningRate = 0.1;
            int maxEpochs = 2000;

            // Create MLP with softmax output activation
            MLP nn = new MLP(NI, NH, NO, ActivationFunctionType.SOFTMAX);

            // Write configuration details to file
            writer.write("Letter Recognition Experiment Results\n");
            writer.write("=====================================\n");
            writer.write("Configuration:\n");
            writer.write("Number of Inputs: " + NI + "\n");
            writer.write("Number of Hidden Units: " + NH + "\n");
            writer.write("Number of Outputs: " + NO + "\n");
            writer.write("Learning Rate: " + learningRate + "\n");
            writer.write("Max Epochs: " + maxEpochs + "\n\n");

            // Training loop
            int batchSize = 10; // Update weights after every 10 examples
            int batchCounter = 0;
            ArrayList<Double> epochErrors = new ArrayList<>();

            for (int epoch = 1; epoch <= maxEpochs; epoch++) {
                double totalError = 0;

                // Shuffle training data for each epoch
                Collections.shuffle(trainingData, new Random(42));

                // Train on each example
                for (TrainingExample example : trainingData) {
                    nn.forward(example.input);
                    totalError += nn.backward(example.input, example.output); // Accumulate updates
                    batchCounter++;

                    // Update weights after every 10 examples
                    if (batchCounter % batchSize == 0) {
                        nn.updateWeights(learningRate);
                    }
                }

                // Apply remaining updates
                if (batchCounter % batchSize != 0) {
                    nn.updateWeights(learningRate);
                }

                batchCounter = 0; // Reset batch counter for next epoch

                // Log error for selected epochs
                if (epoch % 200 == 0 || epoch == maxEpochs) {
                    epochErrors.add(totalError / trainingSamples);
                }
            }

            // Write training error to file
            writer.write("Training Error Over Selected Epochs:\n");
            writer.write("Epoch\tError\n");
            for (int i = 0; i < epochErrors.size(); i++) {
                int epochLogged = (i * 200) + 200; // Every 200 epochs
                writer.write(epochLogged + "\t" + epochErrors.get(i) + "\n");
            }

            // Evaluate on test set
            int correct = 0;
            for (TrainingExample example : testData) {
                nn.forward(example.input);
                int predictedIndex = argMax(nn.O);
                int actualIndex = argMax(example.output);
                if (predictedIndex == actualIndex) {
                    correct++;
                }
            }

            double accuracy = (double) correct / testData.size() * 100;

            // Write final results to file
            writer.write("\nFinal Test Set Accuracy: " + String.format("%.2f", accuracy) + "%\n");
            System.out.println("Results saved to LetterRecognitionExperimentResults.txt");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Loading dataset
    public static ArrayList<TrainingExample> loadDataset(String filename) {
        ArrayList<TrainingExample> dataset = new ArrayList<>();
        HashMap<Character, Integer> letterToIndex = new HashMap<>();
        for (int i = 0; i < 26; i++) {
            letterToIndex.put((char) ('A' + i), i);
        }

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] tokens = line.split(",");
                char letter = tokens[0].charAt(0);
                double[] input = new double[16];
                for (int i = 1; i <= 16; i++) {
                    input[i - 1] = Double.parseDouble(tokens[i]) / 15.0; // Normalize inputs
                }
                double[] output = new double[26];
                int index = letterToIndex.get(letter);
                output[index] = 1.0; // One-hot encoding
                dataset.add(new TrainingExample(input, output));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return dataset;
    }

    // Find index of maximum value in array
    public static int argMax(double[] array) {
        int index = 0;
        double max = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                index = i;
            }
        }
        return index;
    }
}
