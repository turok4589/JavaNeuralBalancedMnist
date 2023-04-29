package neural.project.failedtraining;

import org.encog.mathutil.Equilateral;
import org.encog.ml.data.MLDataSet;
import org.encog.neural.networks.BasicNetwork;

public class MExerciseV3{
    record Report(int tried, int hit) {}
    private final MLDataSet dataset;
    private final BasicNetwork network;

    private int datasize = 0;

    public MExerciseV3(BasicNetwork network, MLDataSet dataset) {
        this.dataset = dataset;
        this.network = network;
        this.datasize = dataset.size();
    }

    public Report report() {
        int tried = dataset.size();
        int hit = 0;
        double digit;
        double label;
        int j = 0;
        for(int i = 0; i < datasize; i++){
            label = dataset.get(i).getIdealArray()[0];
            digit = network.compute(dataset.get(i).getInput()).getData()[0];
            if(j < 40){
                System.out.println("Expected: " + label);
                System.out.println("Network " + digit);
                j++;
            }
            if (label == digit){
                hit++;
            }
        }
        return new Report(tried,hit);
    }
}

