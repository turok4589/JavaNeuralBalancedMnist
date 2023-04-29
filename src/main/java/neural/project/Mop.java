package neural.project;

import java.util.Arrays;

public class Mop implements IMop {

    /**
     * Slice a matrix horizontally
     * The goal of this method is to return rows in a matrix within a certain range.
     * @param src
     * @param startRow
     * @param endRow
     * @return
     */
    public double[][] slice(double[][] src, int startRow, int endRow) {
        //saying src[0].length will get the number of columns.
        double[][] newArr = new double[endRow - startRow][src[0].length];
        int curLoc = 0;
        for (int i = startRow; i < endRow; i++) {
            newArr[curLoc] = src[i];
            curLoc++;
        }
        return newArr;
    }

    /**
     * Flips a matrix to put it into column major order.
     * @param src
     * @return
     */
    public double[][] transpose(double[][] src) {
        double[][] newArr = new double[src[0].length][src.length];
        for (int i = 0; i < src[0].length; i++) {
            for (int j = 0; j < src.length; j++) {
                newArr[i][j] = src[j][i];
            }
        }
        return newArr;
    }

    /**
     * Dice the array vertically
     * @param src
     * @param startCol
     * @param endCol
     * @return
     */
    public double[][] dice(double[][] src, int startCol, int endCol) {
        double[][] newArr = new double[src.length][endCol - startCol];
        // int currentlocation = 0;
        for (int i = 0; i < src.length; i++) {
            //Google what copyOfRange Does
            //For loop starts at 0 because no matter we need to access all the rows
            //to reach the columns
            newArr[i] = Arrays.copyOfRange(src[i],startCol,endCol);
            /**
             * This is the same as the copy of range method
             for(int j = startcol; j < endcol; j++){
                 newarr[i][currentlocation] = src[i][j];
                 currentlocation++;
             **/
        }
        return newArr;
    }

    /**
     * Print mop output
     * @param response
     * @param src
     */
    public void print(String response, double[][] src) {
        System.out.println(response);
        for (double[] x : src) {
            for (double y : x) {
                System.out.print(y + ", ");
            }
            System.out.println();
        }
    }
}
