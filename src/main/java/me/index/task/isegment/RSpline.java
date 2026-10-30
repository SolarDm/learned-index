package me.index.task.isegment;

public enum RSpline implements Window {
    INSTANCE;

    // написано без классов, чтобы лишни й раз не выделять память
    private double startX;
    private double startY;
    private double upX;
    private double upY;
    private double downX;
    private double downY;

    @Override
    public int init_skip(long key, long err) {
        startX = key;
        startY = 0;

        downX = 0.0;
        downY = 0.0;

        upX = 0.0;
        upY = 0.0;

        return 1;
    }

    @Override
    public boolean can_expand(long key, long pos, int err) {
        double newX = key - startX;
        double newUpY = (double) pos + err - startY;
        double newDownY = (double) pos - err - startY;

        if (downX == 0 && downY == 0 && upX == 0 && upY == 0) {
           downY = newDownY;
           downX = newX;

           upY = newUpY;
           upX = newX;

           return true;
        }

        // вообще в статье не совсем так, но не вижу проблем в пересечении именно 
        // интервалов, так как условие на ошибку вполне соблюдается
        if (on_left_side(newX, newDownY, upX, upY)
                || on_left_side(downX, downY, newX, newUpY)) {
            return false;
        }

        if (on_left_side(upX, upY, newX, newUpY)) {
            upX = newX;
            upY = newUpY;
        }

        if (on_left_side(newX, newDownY, downX, downY)) {
            downX = newX;
            downY = newDownY;
        }

        return true;
    }

    @Override
    public double[] get_result() {
        if (upX == 0.0) {
            return new double[]{0.0, startY};
        }

        double minSlope = Math.max(0.0, downY / downX);
        double maxSlope = upY / upX;
        double slope = minSlope + (maxSlope - minSlope) / 2.0;
        double intercept = startY - slope * startX;
        return new double[]{slope, intercept};
    }

    private boolean on_left_side(double ax, double ay, double bx, double by) {
        return ax * by - bx * ay < 0;
    }
}
