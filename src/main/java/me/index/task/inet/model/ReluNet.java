package me.index.task.inet.model;

import me.index.task.inet.Init;
import me.index.task.inet.loss.Loss;

import java.util.Random;

public final class ReluNet extends AbstractNet {
    public ReluNet(int[] layerSizes, double learningRate, int batchSize, Loss loss, Random rnd) {
        super(layerSizes, learningRate, batchSize, loss, rnd, Init.HE, "relu");
    }

    @Override
    protected double activate(double z) {
        return Math.max(0.0, z);
    }

    @Override
    protected double derivative(double z, double value) {
        return z > 0 ? 1.0 : 0.0;
    }
}
