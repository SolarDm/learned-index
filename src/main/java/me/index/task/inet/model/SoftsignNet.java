package me.index.task.inet.model;

import me.index.task.inet.Init;
import me.index.task.inet.loss.Loss;

import java.util.Random;

public final class SoftsignNet extends MiniBatchNet {
    public SoftsignNet(int[] layerSizes, double learningRate, int batchSize, Loss loss, Random rnd) {
        super(layerSizes, learningRate, batchSize, loss, rnd, Init.XAVIER);
    }

    @Override
    protected double activation(double z) {
        return z / (1.0 + Math.abs(z));
    }

    @Override
    protected double derivativeFromActivation(double activation) {
        double d = 1.0 - Math.abs(activation);
        return d * d;
    }

    @Override
    protected String activationName() {
        return "softsign";
    }
}
