package neural.project.failedtraining;

import neural.project.MExcercise;
import neural.project.IMLoader;
import neural.project.IMop;
import neural.project.Mop;
import neural.project.MLoader;
import neural.util.EncogHelper;
import neural.util.IrisHelper;
import org.encog.Encog;
import org.encog.mathutil.Equilateral;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.arrayutil.NormalizedField;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

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
public class ExperimentTesting {
    /**
     * These learning parameters generally give good results according to literature,
     * that is, the training algorithm converges with the tolerance below.
     * */
    public final static double LEARNING_RATE = 0.25;
    public final static double LEARNING_MOMENTUM = 0.25;
    public final static double NORMALIZED_HI = 1;
    public final static double NORMALIZED_LO = -1;

    public final static int NUM_SAMPLES = 2000;
    public static int datasize;
    public final static Map<Integer,NormalizedField> normalizers =
            new HashMap<>();
    public static final Equilateral eq =
            new Equilateral(IrisHelper.species2Cat.size(),
                    NORMALIZED_HI,
                    NORMALIZED_LO);
    private static final DecimalFormat decForm = new DecimalFormat("00.0");
    static SimpleDateFormat dateFormat = new SimpleDateFormat("E MMM dd HH:mm:ss z yyyy");

    /**
     * Initializes the testing matrix's with the mloader
     * @throws IOException
     */
    public static void init() throws IOException {
        //You might need to change these to the appropriate directory.
        String pixelpath = "C:\\Users\\drat6\\Documents\\GitHub\\MiguelVasquez_JavaNeural\\data\\MNISTDATA_Letters\\emnist-balanced-test-images-idx3-ubyte\\";
        String labelpath = "C:\\Users\\drat6\\Documents\\GitHub\\MiguelVasquez_JavaNeural\\data\\MNISTDATA_Letters\\emnist-balanced-test-labels-idx1-ubyte\\";
        MLoader mload = new MLoader(pixelpath, labelpath);
        mload.load();
        // Get mop as we may not use all training data
        IMop mop = new Mop();

        IMLoader.Normal normal = mload.normalize();

        TESTING_INPUTS = mop.slice(normal.pixels(),0, NUM_SAMPLES);
        System.out.println("Training Inputs Length: " + TRAINING_INPUTS[0].length);
        assert(TRAINING_INPUTS[0].length == (28*28));

        TESTING_IDEALS = mop.slice(normal.labels(),0, NUM_SAMPLES);
        System.out.println("Training ideals length: " + TRAINING_IDEALS[0].length);
        assert(TRAINING_IDEALS[0].length == (47-1));

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
        String DIR = "C:\\Users\\drat6\\Documents\\GitHub\\MiguelVasquez_JavaNeural\\data\\";
        init();
        System.out.println("Number of training samples:" + NUM_SAMPLES);
        // Create training observations
        MLDataSet trainingSet = new BasicMLDataSet(TESTING_INPUTS, TESTING_IDEALS);
        String loadmodel = "encogmnist-" + NUM_SAMPLES + ".bin";
        System.out.println("Using this model: " + loadmodel);
        BasicNetwork network =
                (BasicNetwork) EncogDirectoryPersistence.loadObject(new File(DIR + loadmodel));

        EncogHelper.summarize(network);
        datasize = trainingSet.size();
        MExerciseV3 Excercise = new MExerciseV3(network, trainingSet);

        int tried = trainingSet.size();
        int hit = Excercise.report().hit();
        String rate = decForm.format(((float)hit/(float)tried)*100);
        System.out.println("success rate = " + hit + "/" + tried + " (" + rate + "%)");
        System.out.println("finished: " + dateFormat.format(new Date()));
        Encog.getInstance().shutdown();

    }
}

