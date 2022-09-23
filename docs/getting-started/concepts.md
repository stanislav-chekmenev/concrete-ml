# Key Concepts

Concrete-ML is built on top of Concrete-Numpy, which enables Numpy programs to be converted into FHE circuits.

## Lifecycle of a Concrete-ML model

### I. Model Development

1. **Training.** A model is trained using plaintext, non-encrypted, training data.
1. **Quantization.** The model is converted into an integer equivalent using quantization. Concrete-ML performs this step  either during training (Quantization-Aware Training) or after training (Post-Training Quantization), depending on model type. Quantization converts inputs, model weights and all intermediate values of the inference computation to integers.
1. **Simulation** using the **Virtual Library**. Testing FHE models on very large datasets can take a long time. Furthermore, not all models are compatible with FHE constraints out-of-the-box. Simulation allows you to execute a model that was quantized, to measure the accuracy it would have in FHE, but also to determine the modifications required to make it FHE compatible.
1. **Compilation.** Once the model is quantized, it is compiled using Concrete's FHE compiler to produce an equivalent FHE circuit. This circuit is represented as an MLIR program consisting of low level cryptographic operations. You can read more about FHE compilation [here](../advanced-topics/compilation.md), MLIR [here](https://mlir.llvm.org/) and about the low-level Concrete library [here](https://docs.zama.ai/concrete-core).
1. **Inference.** The compiled model can then be executed on encrypted data, once the proper keys have been generated. The model can also be deployed to a server and used to run private inference on encrypted inputs.

You can see some examples of the model development workflow [here](../built-in-models/ml_examples.md).

### II. Model deployment

6. **Client/Server deployment.** In a client/server setting, the model can be exported in a way that:
   - allows the client to generate keys, encrypt and decrypt.
   - provides a compiled model that can run on the server to perform inference on encrypted data
1. **Key generation**. The data owner (client) needs to generate a set of keys to encrypt/decrypt their data, and the model's prediction.

You can see an example of the model deployment workflow [here](https://github.com/zama-ai/concrete-ml/blob/release/0.3.x/docs/advanced_examples/ClientServer.ipynb).

## Cryptography Concepts

Concrete-ML and Concrete-Numpy are tools that hide away the details of the underlying cryptography scheme, TFHE. However, some cryptography concepts are still useful when using these two tool-kits:

1. **Encryption/Decryption**. These operations transform between plaintext, i.e. human-readable information, and  ciphertext, i.e. data that contains a form of the original plaintext that is unreadable by a human or computer without the proper cipher to decrypt it. Encryption takes plaintext and produces ciphertext, while decryption is the inverse operation.
1. **Encrypted inference**. FHE allows a third party to execute (i.e. run inference or predict) a machine learning model on encrypted data (a ciphertext). The result of the inference is also encrypted and can only be read by the person who encrypted the original data.
1. **Keys**. A key is a string of bits used within an encryption algorithm for encrypting data so that the corresponding ciphertext appears random.
1. **Key generation**. Cryptographic keys need to be generated using random number generators. Their size may be large and key generation may take a long time. However, keys only need to be generated once for each model a client uses.
1. **Guaranteed correctness of encrypted computations**. To achieve security, TFHE, the underlying encryption scheme, adds random noise is ciphertexts. This can induce errors during processing of encrypted data, depending on noise parameters. By default, Concrete-ML uses parameters that ensure the correctness of the encrypted computation, so you do not need to take into account the noise parametrization:  results on encrypted data will be the same as the results of simulation on clear data.

While Concrete-ML users only need to understand the cryptography concepts above, for a deeper understanding of the cryptography
behind the Concrete stack, please see the [whitepaper on TFHE and Programmable Boostrapping](https://whitepaper.zama.ai/).

## Model accuracy considerations under FHE constraints

To respect FHE constraints, all numerical programs over encrypted data must have all inputs, constants and intermediate values represented with integers of a maximum of 8 bits.

Thus, Concrete-ML quantizes the input data and model outputs  in the same way as weights and activations. The main levers to control accumulator bit-width are the numbers of bits used for the input, weights and activations of the model. These parameters are crucial to comply with the constraint on accumulator bit-width. Please refer to [the quantization documentation](../advanced-topics/quantization.md) for more details about how to develop models with quantization in Concrete-ML.

However, these methods may cause a reduction of the accuracy of the model since its representation power is diminished. Most importantly, carefully choosing a quantization approach can alleviate accuracy loss, all the while allowing compilation to FHE. Concrete-ML built-in models already include quantization algorithms, and users only need to configure some of their parameters, such as the number of bits discussed above. See [the advanced quantization guide](../advanced-topics/quantization.md#configuring-model-quantization-parameters) for information about configuring these parameters for various models.

Additional specific methods can help to make models compatible with FHE constraints. For instance, dimensionality reduction can reduce the number of input features and thus the maximum accumulator bit-width reached within a circuit. Similarly, sparsity inducing training methods, such as pruning, de-activate some features during inference, which also helps. For now, dimensionality reduction is considered as a pre-processing step, while pruning is used in the [built-in neural networks](../built-in-models/neural-networks.md).

The configuration of model quantization parameters is illustrated in the advanced examples for [Linear and Logistic Regressions](../built-in-models/ml_examples.md) and dimensionality reduction is shown in the [Poisson regression example](https://github.com/zama-ai/concrete-ml/blob/release/0.3.x/docs/advanced_examples/PoissonRegression.ipynb).