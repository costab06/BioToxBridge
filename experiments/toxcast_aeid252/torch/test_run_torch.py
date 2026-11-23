import unittest
from types import SimpleNamespace
import numpy as np
import torch
from run_torch import model_for, fit, predict

class TorchBackendTest(unittest.TestCase):
    def test_probabilities_and_fit(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(100,12)).astype('float32')
        y=(x[:,0]+0.5*x[:,1]>0).astype(int)
        a=SimpleNamespace(dropout=0.1,lr=0.005,weight_decay=0.0001,
                          batch_size=32,epochs=8,min_delta=0.0001,patience=3)
        m,epoch,_=fit(x,y,(24,8),torch.device('cpu'),a,42,
                      np.arange(80),np.arange(80,100))
        p=predict(m,x,torch.device('cpu'),32)
        self.assertEqual(p.shape,(100,))
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(((p>=0)&(p<=1)).all())
        self.assertGreaterEqual(epoch,1)
    def test_logits_not_clipped_or_normalized(self):
        m=model_for(12,(24,8),0.2)
        self.assertIsInstance(m[-1],torch.nn.Linear)

if __name__=='__main__':unittest.main()
