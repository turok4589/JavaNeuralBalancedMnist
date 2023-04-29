package neural.project.failedtraining;

import neural.project.IMLoader;
import neural.project.IMop;
import neural.project.Mop;
import neural.util.EncogHelper;
import neural.util.IrisHelper;
import org.encog.Encog;
import org.encog.engine.network.activation.*;
import org.encog.mathutil.Equilateral;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.BasicTraining;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.arrayutil.NormalizedField;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

import static neural.util.EncogHelper.*;

/**
 * XOR: This example is essentially the "Hello World" of neural network
 * programming. This example shows how to construct an Encog neural network to
 * predict the report from the XOR operator. This example uses backpropagation
 * to train the neural network.
 *
 * This example attempts to use a minimum of Encog values to create and train
 * the neural network. This allows you to see exactly what is going on. For a
 * more advanced example, that uses Encog factories, refer to the XORFactory
 * example.
 *
 * The original version of this code does not appear to converge. I fixed this
 * problem by using two neurons in the hidden layer and instead of ramped activation,
 * sigmoid activation. This makes the network reflect the model in figure 1.1
 * in the book, d. 11. I also added more comments to make the code more explanatory.
 * @author Miguel Vasquez, James Vetro
 * @date 24 Oct 2017
 */
public class ExperimentTraining {
    /**
     * These learning parameters generally give good results according to literature,
     * that is, the training algorithm converges with the tolerance below.
     * */
    public final static double LEARNING_RATE = 0.01;
    public final static double LEARNING_MOMENTUM = 0.01;
    public final static double NORMALIZED_HI = 1;
    public final static double NORMALIZED_LO = -1;

    public final static int NUM_SAMPLES = 5000;
    public static int datasize;
    public final static Map<Integer,NormalizedField> normalizers =
            new HashMap<>();
    public static final Equilateral eq =
            new Equilateral(IrisHelper.species2Cat.size(),
                    NORMALIZED_HI,
                    NORMALIZED_LO);
    private static final DecimalFormat decForm = new DecimalFormat("00.0");
    static SimpleDateFormat dateFormat = new SimpleDateFormat("E MMM dd HH:mm:ss z yyyy");

    public static void init() throws IOException {
        //You might need to change these to the appropriate directory.
        String pixelpath = "C:\\Users\\drat6\\Documents\\GitHub\\MiguelVasquez_JavaNeural\\data\\MNISTDATA_Letters\\emnist-balanced-train-images-idx3-ubyte\\";
        String labelpath = "C:\\Users\\drat6\\Documents\\GitHub\\MiguelVasquez_JavaNeural\\data\\MNISTDATA_Letters\\emnist-balanced-train-labels-idx1-ubyte\\";
        MLoaderV3 mload = new MLoaderV3(pixelpath, labelpath);
        mload.load();
        // Get mop as we may not use all training data
        IMop mop = new Mop();

        IMLoader.Normal normal = mload.normalize();

        TRAINING_INPUTS = mop.slice(normal.pixels(),0, NUM_SAMPLES);
        System.out.println("Training Inputs Length: " + TRAINING_INPUTS[0].length);
        assert(TRAINING_INPUTS[0].length == (28*28));

        TRAINING_IDEALS = mop.slice(normal.labels(),0, NUM_SAMPLES);
        System.out.println("Training ideals length: " + TRAINING_IDEALS[0].length);
        assert(TRAINING_IDEALS[0].length == (1));

    }

    public static void resultsReport(MLDataSet testingSet, BasicNetwork network){
        System.out.println("Network results:");
        System.out.println(String.format("%4s","#") + String.format("%13s", "Ideal") + String.format("%12s", "Actual"));
        int n = 1;
        int errInt = 0;
        for (MLDataPair pair : testingSet) {
            System.out.printf("%4d ",n);
            final MLData inputs = pair.getInput();
            final MLData outputs = network.compute(inputs);
            final MLData ideals = pair.getIdeal();
            final double ideal[] = ideals.getData();
            final double actual[] = outputs.getData();
            int decIdeal = eq.decode(ideal);
            int decActual = eq.decode(actual);
            System.out.printf("%12s", IrisHelper.cat2Species.get(decIdeal));
            System.out.printf("%12s", IrisHelper.cat2Species.get(decActual));
            if (decIdeal != decActual){
                System.out.print(String.format("%8s","MISSED!"));
                errInt += 1;
            }
            System.out.println();
            n += 1;
        }
        System.out.println("...");
        DecimalFormat f = new DecimalFormat("###.#");
        System.out.println(String.format("success rate = " + (30-errInt) + "/30 " + String.format(f.format(((float)(30-errInt)/30) * 100) )) + "%");
    }

    /**
     * Outputs the results of the normalization and denormalization.
     */
    public static void report(String s, double[][] inputs, double [][] ideals) {
        //assumes row major order
        /**
         *  TRAINING_IDEALS and TESTING_IDEALS haven’t yet been constructed
         *  or even initialized so don’t try to output them, not yet.
         *  You work that in the next lab.
         */
        String COL_SHORT_NAMES[] = {"SL",
                "SW",
                "PL",
                "PW"};
        int columnlength = inputs[0].length;
        int rowlength = inputs.length;
        System.out.println("number of columns: " + columnlength);
        System.out.println("number of rows: " + rowlength);
        int currentcolumn = 1;
        System.out.println("---training inputs");
        for (int i = 0; i < columnlength; i++) {
            NormalizedField norm = normalizers.get(i);
            System.out.println(IrisHelper.COL_SHORT_NAMES[i] + ": " + norm.getActualLow() + "->" + norm.getActualHigh());
        }
        System.out.println();
        System.out.print("# ");
        Stream.of(COL_SHORT_NAMES).forEach(name -> System.out.print(name + "         |     "));
        System.out.println();
        for(int j = 0; j < rowlength ; j++){
            System.out.print(j);
            for(int z = 0; z < columnlength; z++){
                NormalizedField norm = normalizers.get(z);
                System.out.print(String.format(" %1.1f -> %2.1f | ", norm.deNormalize(inputs[j][z]), inputs[j][z]));
                //System.out.print(norm.deNormalize(inputs[j][z]) + "->" + inputs[j][z] + "      |      ");
            }
            System.out.println("");
        }

        System.out.println("--- training outputs");
        System.out.println("#     t1       t2        Decoding");
        for(int j = 0; j < ideals.length; j++){
            int x = eq.decode(ideals[j]);
            System.out.print(j);
            System.out.print(String.format("  %2.4f %2.4f %3d", ideals[j][0], ideals[j][1], x));
            if(IrisHelper.species2Cat.get("setosa") == x){
                System.out.println("-> " + "setosa");
            }
            else if(IrisHelper.species2Cat.get("virginica") == x){
                System.out.println("-> " + "virginica");
            }
            else if(IrisHelper.species2Cat.get("versicolor") == x){
                System.out.println("-> " + "versicolor");
            }

        }
    }


    /** Inputs necessary for XOR. */
    public static double TRAINING_INPUTS[][] = {
            {0.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {1.0, 1.0}
    };

    /** Ideals necessary for XOR.*/
    public static double TRAINING_IDEALS[][] = {
            {0.0},
            {1.0},
            {1.0},
            {0.0}};

    /** Inputs necessary for XOR. */
    public static double TESTING_INPUTS[][];

    /** Ideals necessary for XOR.*/
    public static double TESTING_IDEALS[][];

    /**
     * The main method.
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
        //Best tests with 100, 75, 75, 75, 75
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 100));
        //two hidden layers
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 75));

        // Output layer
        network.addLayer(new BasicLayer(new ActivationSoftMax(), false, 1));

        // No more layers to be added
        network.getStructure().finalizeStructure();

        // Randomize the weights
        network.reset();

        EncogHelper.summarize(network);

        // Create training observations
        MLDataSet trainingSet = new BasicMLDataSet(TRAINING_INPUTS, TRAINING_IDEALS);

        // Use a training object for the learning algorithm, backpropagation.
        //final BasicTraining training = new Backpropagation(network, trainingSet,LEARNING_RATE,LEARNING_MOMENTUM);
        final BasicTraining training = new ResilientPropagation(network, trainingSet);

        // Set learning batch size: 0 = batch, 1 = online, n = batch size
        // See org.encog.neural.networks.training.BatchSize
        // train.setBatchSize(0);

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
        //EncogHelper.report(trainingSet, network);
        if(error < minError) {
            System.out.println("Persisting the network");
            //if error is less than the minimum error persist the network
            EncogDirectoryPersistence.saveObject(
                    new File(DIR+"/encogmnist-" + NUM_SAMPLES + ".bin"),network);
        }
        datasize = trainingSet.size();
        System.out.println("Datasize:" + datasize);
        MExerciseV3 Excercise = new MExerciseV3(network, trainingSet);
        System.out.println("Training Samples: " + NUM_SAMPLES);
        int tried = trainingSet.size();
        int hit = Excercise.report().hit();
        String rate = decForm.format(((float)hit/(float)tried)*100);
        System.out.println("success rate = " + hit + "/" + tried + " (" + rate + "%)");
        System.out.println("finished: " + dateFormat.format(new Date()));
        Encog.getInstance().shutdown();

    }
}

