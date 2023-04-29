package neural.project;

import org.encog.mathutil.Equilateral;
import org.encog.ml.data.MLDataSet;
import org.encog.neural.networks.BasicNetwork;


/**
 * This is the MExcercise version of TANH and I mainly used it to test the Symmetrical Elliot Activation Function.
 * Unless you want to try seeing the result of the tan/symmetrical elliot activation function.
 * Author Miguel Vasquez, James Vetro
 */
public class MExcerciseTANH {
    record Report(int tried, int hit) {}
    private final MLDataSet dataset;
    private final BasicNetwork network;

    private int datasize = 0;

    /**
     * Constructor to set the network, and dataset
     * @param network
     * @param dataset
     */
    public MExcerciseTANH(BasicNetwork network, MLDataSet dataset) {
        this.dataset = dataset;
        this.network = network;
        this.datasize = dataset.size();
    }

    /**
     * Compares what the network outputs to the ideal output
     * If they match it's a hit, and this will be used to get a success rate.
     * Normalized between [-1, 1]
     * @return
     */
    public Report report() {
        int tried = dataset.size();
        int hit = 0;
        Equilateral eq = new Equilateral(47, 1, -1);
        double digit;
        double label;
        int j = 0;
        for(int i = 0; i < datasize; i++){
            label = eq.decode(dataset.get(i).getIdealArray());
            digit = eq.decode(network.compute(dataset.get(i).getInput()).getData());
            if (label == digit){
                hit++;
            }
        }
        return new Report(tried,hit);
    }
}

