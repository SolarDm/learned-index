package me.index.task.inet.loss;

public final class HuberLoss implements Loss {
    private final double delta;

    public HuberLoss(double delta) {
        if (!(0 < delta) || !Double.isFinite(delta)) {
            throw new IllegalArgumentException("delta must be finite and positive, got " + delta);
        }
        this.delta = delta;
    }

    @Override
    public double value(double predicted, double target) {
        final double error = predicted - target;
        if (Math.abs(error) <= delta) {
            return error * error / 2;
        }
        return delta * (Math.abs(error) - delta / 2);
    }

    @Override
    public double gradient(double predicted, double target) {
        final double error = predicted - target;
        if (Math.abs(error) <= delta) {
            return error;
        }
        return delta * Math.signum(error);
    }

    @Override
    public String id() {
        return "huber(" + delta + ")";
    }
}
