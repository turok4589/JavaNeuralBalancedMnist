package neural.labs.labs03_06;

import org.encog.util.arrayutil.NormalizationAction;
import org.encog.util.arrayutil.NormalizedField;

public class SimpleNormalize {
    public static void main(String[] args) {
        //Exam Question
        //Suppose you have the normalization equation in Figure 3.
        // The unnormalized, "real world" range is 0.10 - 2.50 and
        // the corresponding normalized range for MLP input is -1.0 to 1.0.
        //What is the normalized value of 1.80 rounded to exactly two decimal places?
        //Answer 0.42 rounded to decimal places
        NormalizedField norm = new NormalizedField(NormalizationAction.Normalize,
                null,9,0,1,0);

        double x = 7;
        double y = norm.normalize(x);

        System.out.println( x + " normalized is " + y);

        double z = norm.deNormalize(y);

        System.out.println( y + " denormalized is " + z);
    }
}