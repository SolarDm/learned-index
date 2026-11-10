package me.index.task.inet.model;

import me.index.task.inet.Init;
import me.index.task.inet.loss.Loss;

import java.util.Random;

public final class LeakyReluNet extends MiniBatchNet {
    private static final double SLOPE = 0.01;

    public LeakyReluNet(int[] layerSizes, double learningRate, int batchSize, Loss loss, Random rnd) {
        super(layerSizes, learningRate, batchSize, loss, rnd, Init.HE);
    }

    @Override
    protected double activation(double z) {
        return z > 0 ? z : SLOPE * z;
    }

    @Override
    protected double derivativeFromActivation(double activation) {
        return activation > 0 ? 1.0 : SLOPE;
    }

    @Override
    protected String activationName() {
        return "leaky_relu";
    }
}
