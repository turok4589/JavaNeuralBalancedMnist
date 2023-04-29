package neural.labs.labs03_06;

import neural.project.IMop;
import neural.project.Mop;
import neural.util.EncogHelper;
import neural.util.IrisHelper;
import org.apache.commons.math3.stat.StatUtils;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.mathutil.Equilateral;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.BasicTraining;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.util.arrayutil.NormalizationAction;
import org.encog.util.arrayutil.NormalizedField;

import java.text.DecimalFormat;
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
 * @author Ron Coleman
 * @date 24 Oct 2017
 */
public class JamesVetroZiris {
    /**
     * These learning parameters generally give good results according to literature,
     * that is, the training algorithm converges with the tolerance below.
     * */
    public final static double LEARNING_RATE = 0.25;
    public final static double LEARNING_MOMENTUM = 0.25;
    public final static double NORMALIZED_HI = 1;
    public final static double NORMALIZED_LO = -1;
    public final static Map<Integer,NormalizedField> normalizers =
            new HashMap<>();
    public static final Equilateral eq =
            new Equilateral(IrisHelper.species2Cat.size(),
                    NORMALIZED_HI,
                    NORMALIZED_LO);


    /**
     * @param src is a 2d array in row major order
     * @newaray that is in row major order and the data is normalized.
     */
    public static double [][] normalize(double [][] src){
        //double[][] flippedarray = new double[src[0].length][src.length];
        IMop mop = new Mop();
        double [][] dice;
        double [][] transpose;
        //Get a column. Use IMP to dice then transpose the src matrix.
        int columnlength  = src[0].length;
        //need every column
        int currentcolumn = 1;
        for(int i = 0; i < columnlength; i++){
            transpose = mop.transpose(mop.dice(src,0, currentcolumn));
            //diced and transposed a column.
            //Calculate the column maximum and minimum ranges.
            //Use the StatUtils max and min methods, respectively.
            double max = StatUtils.max(transpose[i]);
            double min = StatUtils.min(transpose[i]);
            /**
             * Construct a NormalizedField instance for each
             * column using the max/min column ranges
             * and use NormalizedField to normalize every datum in the src data.
             * Be careful to give the max/min and hi/lo parameters in the correct order.
             * See SimpleNormalize.java for how to do this.
             */
            NormalizedField norm = new NormalizedField(NormalizationAction.Normalize,
                    "columns", max, min, NORMALIZED_HI, NORMALIZED_LO);
            //need to normalize the column in the src[][]
            //which is the row in transpose
            for(int j = 0; j < transpose[0].length; j++){
                //System.out.println("Original: " + src[j][i]);
                //System.out.println("Normalize: " + norm.normalize(src[j][i]));
                src[j][i] = norm.normalize(src[j][i]);
                //System.out.println("Denormalize: " + norm.deNormalize(flippedarray[i][j]));
            }
            //System.out.println("Loop Broke");
            //cache the normalizedfield in the hashmap
            normalizers.put(i, norm);
            currentcolumn++;
        }
        //returns a matrix in row major order
        //so in this case we return the normalized src matrix.
        return src;
    }

    public static double[][] encode(double [][] src){
        IMop mop = new Mop();
        double [][] dice;
        double [][] transpose;
        double [][] encodedarray = new double[150][IrisHelper.species2Cat.size() - 1];
        //Get a column. Use IMP to dice then transpose the src matrix.
        int columnlength  = src[0].length;
        int currentcolumn = 1;
        transpose = mop.transpose(mop.dice(src,0, currentcolumn));
        //Calculate the column maximum and minimum ranges.
        //Use the StatUtils max and min methods, respectively.
        double max = StatUtils.max(transpose[0]);
        double min = StatUtils.min(transpose[0]);
        //Now it's 1*150 matrix. So access the first column with
        for(int j = 0; j < transpose[0].length; j++) {
            //encode returns an array based on the set.
            encodedarray[j] = eq.encode((int) src[j][0]);
            //System.out.println("J Level: " + j);
            //System.out.println(encodedarray[j][0]);
            //System.out.println(encodedarray[j][1]);
            //System.out.println(eq.decode(encodedarray[j]));
            //System.out.println(eq.getDistance(encodedarray[j], (int)src[j][0]));
        }

        //returns a matrix in row major order
        //so in this case we return the normalized src matrix.
        return encodedarray;
    }

    public static void init(){
        //observations is a matrix in row major order.
        IMop mop = new Mop();
        double[][] observations = mop.transpose(IrisHelper.load("data/iris.csv"));
        double[][] observations_ = mop.dice(observations,0,4);
        double[][] inputs = normalize(observations_);
        //slice the normalized inputs for training and testing
        observations_ = mop.dice(observations,4,5);
        double[][] outputs = encode(observations_);
        TRAINING_INPUTS = mop.slice(inputs,0,120);
        TESTING_INPUTS = mop.slice(inputs, 120, 150);


        TRAINING_IDEALS = mop.slice(outputs,0,120);
        TESTING_IDEALS = mop.slice(outputs,120,150);
        System.out.println("Training");
        report("training", TRAINING_INPUTS, TRAINING_IDEALS);
        System.out.println();
        System.out.println("Testing");
        report("testing",TESTING_INPUTS, TESTING_IDEALS);
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
         *  TRAINING_IDEALS and TESTING_IDEALS havenâ€™t yet been constructed
         *  or even initialized so donâ€™t try to output them, not yet.
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
    public static void main(final String args[]) {
        init();

        // Instantiate the network
        BasicNetwork network = new BasicNetwork();

        // Input layer plus bias node
        network.addLayer(new BasicLayer(null, true, 4));

        // Hidden layer plus bias node
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 4));

        // Output layer
        network.addLayer(new BasicLayer(new ActivationTANH(), false, 2));

        // No more layers to be added
        network.getStructure().finalizeStructure();

        // Randomize the weights
        network.reset();

        EncogHelper.describe(network);

        // Create training observations
        MLDataSet trainingSet = new BasicMLDataSet(TRAINING_INPUTS, TRAINING_IDEALS);

        // Use a training object for the learning algorithm, backpropagation.
//        final BasicTraining training = new Backpropagation(network, trainingSet,LEARNING_RATE,LEARNING_MOMENTUM);
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
            }
            else
                sameCount++;

            if(sameCount > MAX_SAME_COUNT)
                break;

            EncogHelper.log(epoch, error,false,false);

        } while (error > TOLERANCE && epoch < MAX_EPOCHS);

        training.finishTraining();

        EncogHelper.log(epoch, error,sameCount > MAX_SAME_COUNT, true);
        EncogHelper.report(trainingSet, network);
        EncogHelper.describe(network);

        MLDataSet testSet = new BasicMLDataSet(TESTING_INPUTS,TESTING_IDEALS);
        resultsReport(testSet, network);
        Encog.getInstance().shutdown();

    }
}
