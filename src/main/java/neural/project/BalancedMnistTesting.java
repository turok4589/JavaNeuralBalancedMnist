package neural.project;

import neural.util.EncogHelper;
import org.encog.Encog;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.persist.EncogDirectoryPersistence;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Author Miguel Vasquez, James Vetro
 */
public class BalancedMnistTesting {
    public final static int NUM_SAMPLES = 6000;
    public static int datasize;
    private static final DecimalFormat decForm = new DecimalFormat("00.0");
    static SimpleDateFormat dateFormat = new SimpleDateFormat("E MMM dd HH:mm:ss z yyyy");

    /**
     * Initializes MNIST DATA for testing
     * Remember to change path if needed
     * @throws IOException
     */
    public static void init() throws IOException {
        //You might need to change these to the appropriate directory.
        String pixelpath = "C:\\Users\\drat6\\Documents\\GitHub\\MiguelVasquez_JavaNeural\\data\\MNISTDATA_Letters\\emnist-balanced-test-images-idx3-ubyte\\";
        String labelpath = "C:\\Users\\drat6\\Documents\\GitHub\\MiguelVasquez_JavaNeural\\data\\MNISTDATA_Letters\\emnist-balanced-test-labels-idx1-ubyte\\";
        //MLoader mload = new MLoader(pixelpath, labelpath);
        MLoaderTANH mload = new MLoaderTANH(pixelpath, labelpath);
        mload.load();

        IMLoader.Normal normal = mload.normalize();

        TESTING_INPUTS = normal.pixels();
        assert(TESTING_INPUTS[0].length == 784);

        TESTING_IDEALS = normal.labels();
        assert(TESTING_IDEALS[0].length == (47-1));

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
     * Grabs trained network, and tests with the testing set.
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
        MExcerciseTANH Excercise = new MExcerciseTANH(network, trainingSet);
        //MExcercise Excercise = new MExcercise(network, trainingSet);

        int tried = trainingSet.size();
        int hit = Excercise.report().hit();
        String rate = decForm.format(((float)hit/(float)tried)*100);
        System.out.println("success rate = " + hit + "/" + tried + " (" + rate + "%)");
        System.out.println("finished: " + dateFormat.format(new Date()));
        Encog.getInstance().shutdown();

    }
}

