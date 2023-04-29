package neural.project;

import org.apache.commons.math3.stat.StatUtils;
import org.encog.mathutil.Equilateral;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.zip.CRC32;
import org.encog.util.arrayutil.NormalizationAction;
import org.encog.util.arrayutil.NormalizedField;

/**
 * The purpose of this MLoader is to normalize the pixels so that it's between -1, 1
 * This can be ignored if you don't want to test the network using the tan activation function.
 */
public class MLoaderTANH implements IMLoader {
    CRC32 crc = new CRC32();
    private int magicnumber = 0;
    public int numberofitems = 0;
    private int rows = 0;
    private int columns = 0;

    private int rowpluscolumn = 0;
    private int labelnumber = 0;
    private int numberoflabels = 0;
    private int labeloutput = 0;
    private int currentstream = 0;
    public double[] mdigitdata;

    public final static double NORMALIZED_HI = 1;
    public final static double NORMALIZED_LO = -1;
    MDigit currentpixel = null;

    MDigit[] loaddata = null;


    private String pixelpath;
    private String labelpath;

    public final static int numDigitsAndLetters = 47;
    public static final Equilateral eq =
            new Equilateral(numDigitsAndLetters,
                    NORMALIZED_HI,
                    NORMALIZED_LO);

    public final static Map<Integer,NormalizedField> normalizers =
            new HashMap<>();

    /**
     * Construct to set the paths
     * @param pixelpaths
     * @param labelpaths
     */
    public MLoaderTANH(String pixelpaths, String labelpaths){
        pixelpath = pixelpaths;
        labelpath = labelpaths;
    }

    /**
     * Loads The MNIST data and places it inside an array of MDigit objects.
     * @return MDigit[]
     * @throws IOException
     */
    @Override
    public MDigit[] load() throws IOException {
        double max = 0.0;
        IMop mop = new Mop();
        DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(pixelpath)));
        magicnumber = dis.readInt();
        numberofitems = dis.readInt();
        rows = dis.readInt();
        columns = dis.readInt();
        rowpluscolumn = rows*columns;
        DataInputStream labels = new DataInputStream(new BufferedInputStream(new FileInputStream(labelpath)));
        labelnumber = labels.readInt();
        numberoflabels = labels.readInt();
        loaddata = new MDigit[numberofitems];
        System.out.println("Row and columns length " + (rows*columns));

        if(numberofitems == numberoflabels) {
            for (int i = 0; i < numberoflabels; i++) {
                //need a new array to fit in all the pixels each time
                mdigitdata = new double[rows * columns];
                int lableoutput = labels.readUnsignedByte();
                for (int j = 0; j < rows * columns; j++) {
                    currentstream = dis.readUnsignedByte();
                    if(max < currentstream){ max = currentstream;}
                    crc.update(currentstream);
                    mdigitdata[j] = currentstream;
                }
                currentpixel = new MDigit(i, mdigitdata, lableoutput);
                loaddata[i] = currentpixel;
            }
        }
        System.out.println("Max Data " + max);
        return loaddata;
    }

    /**
     *{@inheritDoc}
     */
    @Override
    public int getPixelsMagic() {
        return magicnumber;
    }

    /**
     *{@inheritDoc}
     */
    @Override
    public int getLabelsMagic() {
        return labelnumber;
    }

    /**
     *{@inheritDoc}
     */
    @Override
    public long getChecksum() {
        return crc.getValue();
    }

    /**
     * Normalizes the pixels by using the normalizer to put them between -1 and 1.
     * Labels are put into the equilateral encoder
     * @return Normal Object Containing two Matrices
     */
    @Override
    public Normal normalize() {
        //this is going to use the load data array
        double[][] newpixel = new double[numberofitems][rowpluscolumn];
        //46 because 47 categories = 46 dimensional vector
        double[][] encodedlables = new double[numberofitems][46];
        for (int i = 0; i < loaddata.length; i++) {
            //going to normalize and encode the pixels and labels respectively
            MDigit n = loaddata[i];
            double max = (double) StatUtils.max(n.pixels());
            double min = (double) StatUtils.min(n.pixels());
            NormalizedField norm = new NormalizedField(NormalizationAction.Normalize,
                    "pixels", max, min, NORMALIZED_HI, NORMALIZED_LO);
            for (int j = 0; j < n.pixels().length; j++) {
                newpixel[i][j] = norm.normalize(n.pixels()[j]);
            }

            encodedlables[i] = eq.encode(n.label());
        }
        return new Normal(newpixel, encodedlables);
    }
}
