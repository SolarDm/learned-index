package me.index.task.inet.model;

import me.index.task.inet.Init;
import me.index.task.inet.loss.Loss;

import java.util.Random;

public final class SoftsignNet extends AbstractNet {
    public SoftsignNet(int[] layerSizes, double learningRate, int batchSize, Loss loss, Random rnd) {
        super(layerSizes, learningRate, batchSize, loss, rnd, Init.XAVIER, "softsign");
    }

    @Override
    protected double activate(double z) {
        return z / (1.0 + Math.abs(z));
    }

    @Override
    protected double derivative(double z, double value) {
        double denominator = 1.0 + Math.abs(z);
        return 1.0 / (denominator * denominator);
    }
}
