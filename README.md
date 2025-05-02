<h1>CIFAR-10 Object Recognition using ResNet50</h1>

<h2>📌 Objective</h2>
<p>To build a deep learning model that accurately classifies images from the CIFAR-10 dataset using transfer learning with the ResNet50 architecture.</p>

<hr>

<h2>🛑 Problem Statement</h2>
<p>The CIFAR-10 dataset is a standard benchmark in computer vision. It consists of small 32x32 pixel color images across 10 classes. Training deep convolutional neural networks on such low-resolution data can be challenging. This project leverages transfer learning using the powerful ResNet50 model to improve classification accuracy.</p>

<hr>

<h2>📊 Dataset</h2>
<ul>
  <li><b>Source:</b> <a href="https://www.kaggle.com/c/cifar-10/data" target="_blank">Kaggle - CIFAR-10 Image Classification Challenge</a></li>
  <li><b>Total Images:</b> 60,000</li>
  <li><b>Training Set:</b> 50,000 images</li>
  <li><b>Test Set:</b> 10,000 images</li>
  <li><b>Categories:</b> Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck</li>
</ul>

<hr>

<h2>⚙️ Methodology</h2>

<h3>1. Data Preprocessing</h3>
<ul>
  <li>Resize CIFAR-10 images from 32x32 to 224x224 to match ResNet50 input size</li>
  <li>Normalize pixel values to range [0, 1]</li>
  <li>Apply data augmentation (rotation, flip, zoom) to increase generalization</li>
</ul>

<h3>2. Model Architecture</h3>
<ul>
  <li>Base Model: <b>ResNet50</b> pretrained on ImageNet</li>
  <li>Freeze convolutional base layers initially</li>
  <li>Custom classification head:
    <ul>
      <li>Global Average Pooling</li>
      <li>Dense layers with ReLU</li>
      <li>Final Dense layer with Softmax activation (10 classes)</li>
    </ul>
  </li>
</ul>

<h3>3. Training</h3>
<ul>
  <li>Loss Function: Categorical Crossentropy</li>
  <li>Optimizer: Adam</li>
  <li>Metrics: Accuracy</li>
  <li>EarlyStopping and ReduceLROnPlateau callbacks used</li>
</ul>

<h3>4. Evaluation</h3>
<ul>
  <li>Achieved 94% test accuracy</li>
  <li>Visualized predictions and confusion matrix for model analysis</li>
</ul>

<hr>

<h2>🛠️ Technologies Used</h2>
<ul>
  <li>Python</li>
  <li>TensorFlow / Keras</li>
  <li>NumPy, Pandas</li>
  <li>Matplotlib, Seaborn</li>
</ul>

<hr>

<h2>✅ Results</h2>
<ul>
  <li>ResNet50 achieved strong performance on a low-resolution image dataset</li>
  <li>Transfer learning enabled fast convergence and high accuracy</li>
</ul>

<hr>

<h2>📝 Conclusion</h2>
<p>This project demonstrates how deep residual networks like ResNet50, when fine-tuned appropriately, can significantly improve classification performance even on small-resolution datasets like CIFAR-10.</p>

<hr>

<h2>🔮 Future Enhancements</h2>
<ul>
  <li>Try other pretrained models such as EfficientNet or MobileNet</li>
  <li>Use Grad-CAM for model explainability</li>
  <li>Deploy the model in a web application using Streamlit or Flask</li>
</ul>
