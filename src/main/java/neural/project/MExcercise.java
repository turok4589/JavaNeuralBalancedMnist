package neural.project;

import org.encog.mathutil.Equilateral;
import org.encog.ml.data.MLDataSet;
import org.encog.neural.networks.BasicNetwork;

/**
 * See's if network can match the ideal training set
 * Author Miguel A Vasquez, James Vetro
 */
public class MExcercise {
    record Report(int tried, int hit) {}
    private final MLDataSet dataset;
    private final BasicNetwork network;

    private int datasize = 0;

    /**
     * Constructor to set the network, and dataset
     * @param network
     * @param dataset
     */
    public MExcercise(BasicNetwork network, MLDataSet dataset) {
        this.dataset = dataset;
        this.network = network;
        this.datasize = dataset.size();
    }

    /**
     * Compares what the network outputs to the ideal output
     * If they match it's a hit, and this will be used to get a success rate.
     * @return
     */
    public Report report() {
        int tried = dataset.size();
        int hit = 0;
        Equilateral eq = new Equilateral(47, 1, 0);
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

