package neural.project;
import neural.util.EncogHelper;
import org.encog.Encog;
import org.encog.engine.network.activation.*;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.BasicTraining;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.persist.EncogDirectoryPersistence;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

import static neural.util.EncogHelper.*;

/**
 * Author Miguel Vasquez, James Vetro
 */
public class BalancedMnistTraining {
    public final static int NUM_SAMPLES = 1000;
    public static int datasize;
    private static final DecimalFormat decForm = new DecimalFormat("00.0");
    static SimpleDateFormat dateFormat = new SimpleDateFormat("E MMM dd HH:mm:ss z yyyy");


    /**
     * Uses MLoader to load MNIST DATA, and normalize it
     * Remember to change path if needed.
     * @throws IOException
     */
    public static void init() throws IOException {
        //You might need to change these to the appropriate directory.
        String pixelpath = "C:\\Users\\drat6\\Documents\\GitHub\\MiguelVasquez_JavaNeural\\data\\MNISTDATA_Letters\\emnist-balanced-train-images-idx3-ubyte\\";
        String labelpath = "C:\\Users\\drat6\\Documents\\GitHub\\MiguelVasquez_JavaNeural\\data\\MNISTDATA_Letters\\emnist-balanced-train-labels-idx1-ubyte\\";
        //MLoader mload = new MLoader(pixelpath, labelpath);
        MLoaderTANH mload = new MLoaderTANH(pixelpath, labelpath);
        mload.load();
        // Get mop as we may not use all training data
        IMop mop = new Mop();

        IMLoader.Normal normal = mload.normalize();

        TRAINING_INPUTS = mop.slice(normal.pixels(),0, NUM_SAMPLES);
        System.out.println("Training Inputs Length: " + TRAINING_INPUTS[0].length);
        assert(TRAINING_INPUTS[0].length == (28*28));

        TRAINING_IDEALS = mop.slice(normal.labels(),0, NUM_SAMPLES);
        System.out.println("Training ideals length: " + TRAINING_IDEALS[0].length);
        assert(TRAINING_IDEALS[0].length == (47-1));

    }

    /** Inputs necessary for MNIST. */
    public static double TRAINING_INPUTS[][];

    /** Ideals necessary for MNIST.*/
    public static double TRAINING_IDEALS[][];

    /** Inputs necessary for MNIST. */
    public static double TESTING_INPUTS[][];

    /** Ideals necessary for MNIST.*/
    public static double TESTING_IDEALS[][];

    /**
     * The main method.
     * Builds network, and does training.
     * @param args No arguments are used.
     */
    public static void main(final String args[]) throws IOException {
        String DIR = "C:\\Users\\drat6\\Documents\\GitHub\\MiguelVasquez_JavaNeural\\data";

        init();

        // Instantiate the network
        System.out.println("Started: "+dateFormat.format(new Date()));
        BasicNetwork network = new BasicNetwork();

        // Input layer plus bias node
        network.addLayer(new BasicLayer(null, true, 784));

        // Hidden layer plus bias node

        /**
         * Figured if sigmoid isn't working tanh would be a good alternative.
         * ElliottSymmetric is more efficient TANH according to encog
         * Produced best results, but it takes too long to converge.
         * You also need to change the Mloader to MloaderTANH, and the MExcercise to MExcerciseTANH
         * Change Tolerance to 0.016 just to speed things up. It will eventually reach the 0.01 tolerance but takes a while
         * There is also a risk of overfitting as 1 test saw a 100% success rate so that's another reason I chose to increase the tolerance.
         * Below is 1 hidden layer, and 1 output layer
         */
        network.addLayer(new BasicLayer(new ActivationElliottSymmetric(), true, 100));

        network.addLayer(new BasicLayer(new ActivationElliottSymmetric(), false, 46));


        /**
         * Second best results came from using ActivationElliott() with 700 neurons
         * However, this takes an extremely long time to run.
         * If you simply want to see if the code runs use a smaller neuron count.
         * 400 Neurons produces okay results better than sigmoid.
         * Below is 1 hidden layer, and 1 output layer
         */
        //network.addLayer(new BasicLayer(new ActivationElliott(), true, 700));
        //network.addLayer(new BasicLayer(new ActivationElliott(), false, 46));

        /**
         * To quickly run the network just use sigmoid with 100 neurons
         * This is a lot worse, but it's better than the rest of my attempts
         * Remember to comment these out if you want to use Elliott
         * Below is 1 hidden layer, and 1 output layer
         */
        //network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 100));

        //network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 46));

        /**
         * I saw some success with this, but this isn't optimal takes way too long to converge.
         * Below is 1 hidden layer, and 1 output layer
         */
        //network.addLayer(new BasicLayer(new ActivationBipolarSteepenedSigmoid(), true, 400));

        //network.addLayer(new BasicLayer(new ActivationBipolarSteepenedSigmoid(), false, 46));

        // No more layers to be added
        network.getStructure().finalizeStructure();

        // Randomize the weights
        network.reset();

        EncogHelper.summarize(network);

        // Create training observations
        MLDataSet trainingSet = new BasicMLDataSet(TRAINING_INPUTS, TRAINING_IDEALS);

        // Use a training object for the learning algorithm, backpropagation.
        final BasicTraining training = new ResilientPropagation(network, trainingSet);

        int epoch = 0;

        double minError = Double.MAX_VALUE;

        double error = 0.0;

        int sameCount = 0;
        final int MAX_SAME_COUNT = 5*LOG_FREQUENCY;

        EncogHelper.log(epoch, error,false, false);
        do {
            training.iteration();

            epoch++;

            error = training.getError();

            if(error < minError) {
                minError = error;
                sameCount = 1;
                //if error is less than the minimum error persist the network
                EncogDirectoryPersistence.saveObject(
                        new File(DIR+"/encogmnist-" + NUM_SAMPLES + ".bin"),network);
            }
            else
                sameCount++;

            if(sameCount > MAX_SAME_COUNT)
                break;

            EncogHelper.log(epoch, error,false,false);

        } while (error > TOLERANCE && epoch < MAX_EPOCHS);

        training.finishTraining();
        EncogHelper.log(epoch, error,sameCount > MAX_SAME_COUNT, true);
        EncogHelper.summarize(network);
        if(error < minError) {
            System.out.println("Persisting the network");
            //if error is less than the minimum error persist the network
            EncogDirectoryPersistence.saveObject(
                    new File(DIR+"/encogmnist-" + NUM_SAMPLES + ".bin"),network);
        }
        datasize = trainingSet.size();
        System.out.println("Datasize:" + datasize);

        /**
         * Use MExcercise for most of the activation functions
         * MExcerciseTANH is solely for tanh, and Symmetric Elliot.
         */
        //MExcercise Excercise = new MExcercise(network, trainingSet);
        MExcerciseTANH Excercise = new MExcerciseTANH(network, trainingSet);
        System.out.println("Training Samples: " + NUM_SAMPLES);
        int tried = trainingSet.size();
        int hit = Excercise.report().hit();
        String rate = decForm.format(((float)hit/(float)tried)*100);
        System.out.println("success rate = " + hit + "/" + tried + " (" + rate + "%)");
        System.out.println("finished: " + dateFormat.format(new Date()));
        Encog.getInstance().shutdown();

    }
}

