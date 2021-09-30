# Handwritten Digit Generator
Artificial Generator that generates handwritten digits. <br>

## Description
This app uses a **cGAN (Conditional Generative Adversarial Networks)** model to generate realistic digit drawings. <br>

## Model
**cGAN** model consists of a **generator** and a **discriminator**. <br>

Comparison of GAN and cGAN architectures: <br>


cGAN structure has an additional input called "label" that is an input for both generator and discriminator. <br>
Using this additional condition, cGAN model can generate images for different types. As an example; we can give digit "7" as a condition
and cGAN can generate drawings of only digit "7". <br>

### Generator Architecture


### Discriminator Architecture


## Usage
Go to [app link](https://handwritten-digit-generator.herokuapp.com/) to test this app. <br>

1. Type the digit you want to generate. (You can leave it empty to generate random digit(s))
2. Type the number of digits to generate. (You can leave it empty to generate just 1 digit)
3. Click "Generate" button.

## Examples
Generated "4" digits: <br>
![generated "4" digits](https://raw.githubusercontent.com/yigitatesh/handwritten_digit_generator_web_app/main/results/generated_4_digits.PNG)

Generated random digits (1): <br>
![generated random digits 1](https://raw.githubusercontent.com/yigitatesh/handwritten_digit_generator_web_app/main/results/generated_random_digits.PNG)

Generated random digits (2): <br>
![generated random digits 2](https://raw.githubusercontent.com/yigitatesh/handwritten_digit_generator_web_app/main/results/generated_random_digits_2.PNG)
