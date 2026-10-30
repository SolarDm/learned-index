package me.index.task.inet.model;

import me.index.task.inet.Init;
import me.index.task.inet.loss.Loss;

import java.util.Random;

public final class LeakyReluNet extends AbstractNet {
    public LeakyReluNet(int[] layerSizes, double learningRate, int batchSize, Loss loss, Random rnd) {
        super(layerSizes, learningRate, batchSize, loss, rnd, Init.HE, "leaky_relu");
    }

    @Override
    protected double activate(double z) {
        return z > 0 ? z : 0.01 * z;
    }

    @Override
    protected double derivative(double z, double value) {
        return z > 0 ? 1.0 : 0.01;
    }
}
