package neural.project.failedtraining;
import neural.project.*;
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

public class MLoaderV3 implements IMLoader {
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

    public MLoaderV3(String pixelpaths, String labelpaths){
        pixelpath = pixelpaths;
        labelpath = labelpaths;
    }
    @Override
    //add try catch for file not found exception later
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

    @Override
    public int getPixelsMagic() {
        return magicnumber;
    }

    @Override
    public int getLabelsMagic() {
        return labelnumber;
    }

    @Override
    public long getChecksum() {
        return crc.getValue();
    }

    @Override
    public Normal normalize() {
        //this is going to use the load data array
        double[][] newpixel = new double[numberofitems][rowpluscolumn];
        //9 because 10 categories = 9 dimensional vector
        double [][] encodedlables = new double[numberofitems][1];
        for(int i = 0; i < loaddata.length; i++){
            //going to normalize and encode the pixels and labels respectively
            MDigit n = loaddata[i];
            for(int j = 0; j < n.pixels().length; j++){
                newpixel[i][j] = (double)n.pixels()[j]/255.0;
            }

            encodedlables[i][0] = n.label();
        }
        return new Normal(newpixel, encodedlables);
    }
    public static void main(String [] args) throws IOException {
        //You might need to change these to the appropriate directory.
        String pixelpath = "C:\\Users\\drat6\\Documents\\GitHub\\MiguelVasquez_JavaNeural\\data\\MNISTDATA_Letters\\emnist-balanced-train-images-idx3-ubyte\\";
        String labelpath = "C:\\Users\\drat6\\Documents\\GitHub\\MiguelVasquez_JavaNeural\\data\\MNISTDATA_Letters\\emnist-balanced-train-labels-idx1-ubyte\\";
        MLoader m = new MLoader(pixelpath, labelpath);
        MDigit[] mnistmatrix = m.load();
        System.out.println(mnistmatrix.length);
        System.out.println(m.numberofitems);
        System.out.println("Pixels magic number " + m.getPixelsMagic());
        System.out.println("Labels magic number " + m.getLabelsMagic());
        MDigit n = mnistmatrix[2];
        double[] pixels = n.pixels();
        double [][] placeholder = new double[28][28];
        int i = 0;
        int j = 0;
        for(int x = 0; x < 28; x++){
            for(int z = 0; z < 28; z++) {
                if(pixels[i] == 0.0){
                    System.out.print(".");
                    i++;
                }
                else {
                    System.out.print((int)pixels[i]);
                    i++;
                }
            }
            System.out.println();
        }
        System.out.print(n);
    }

}
