import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Layer, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import metrics, optimizers

import seaborn as sns
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests



gene_correlation = 0.8
latent_dim = 5
epochs = 50
batch_size = 50
learning_rate = 0.001

activation = 'sigmoid'





x = pd.read_table('./Data/Ov_Cancer_mRNA.txt', index_col=0)
starting_genes = np.load('./Data/Lists/gene_list_'+str(gene_correlation)+'.npy')
x = mRNA_df.iloc[likely_gene_index].T 
x = (x - np.min(x))/(np.max(x) - np.min(x))

train_percent = 0.6
x_train = x.head(int((train_percent)*len(x)))
x_test = x.tail(len(x)-int((train_percent)*len(x)))
original_dim = x.shape[1]
print(x_train.head(5)) #patients go down the left, genes run across




#Set hyperparameters for training. kappa is a variable which adapts the VAE loss function, slowly blending in the KL divergence loss into the reconstruction loss. kappa = 1 corresponds to a full VAE. 
# Initialise variables and hyper parameters
original_dim = x.shape[1]



def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0.,
                              stddev=1)
    z = z_mean + K.exp(z_log_var / 2) * epsilon
    return z


class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        # https://keras.io/layers/writing-your-own-keras-layers/
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x_input, x_decoded):
        reconstruction_loss = original_dim * metrics.binary_crossentropy(x_input, x_decoded)
        kl_loss = - 0.5 * K.sum(1 + z_log_var_encoded - K.square(z_mean_encoded) - 
                                K.exp(z_log_var_encoded), axis=-1)
        return K.mean(reconstruction_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded = inputs[1]
        loss = self.vae_loss(x, x_decoded)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x

          
rnaseq_input = Input(shape=(original_dim, ))

z_mean_dense_linear = Dense(latent_dim, kernel_initializer='glorot_uniform')(rnaseq_input)
z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense_linear)
z_mean_encoded = Activation(activation)(z_mean_dense_batchnorm)

z_log_var_dense_linear = Dense(latent_dim, kernel_initializer='glorot_uniform')(rnaseq_input)
z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense_linear)
z_log_var_encoded = Activation(activation)(z_log_var_dense_batchnorm)

z = Lambda(sampling, output_shape=(latent_dim, ))([z_mean_encoded, z_log_var_encoded])




decoder_to_reconstruct = Dense(original_dim, kernel_initializer='glorot_uniform', activation='sigmoid')
rnaseq_reconstruct = decoder_to_reconstruct(z)






adam = optimizers.Adam(lr=learning_rate)
vae_layer = CustomVariationalLayer()([rnaseq_input, rnaseq_reconstruct])
vae = Model(rnaseq_input, vae_layer)
vae.compile(optimizer=adam, loss=None)




#Train the model. The training data is shuffled after every epoch and 10% of the data is heldout for calculating validation loss.
hist = vae.fit(np.array(x_train),
               shuffle=True,
               epochs=epochs,
               verbose=1,
               batch_size=batch_size,
               validation_data=(np.array(x_test), None),
               callbacks=[])



plt.subplots()
plt.plot(hist.history["val_loss"], color='r', label='Validation loss')
plt.plot(hist.history["loss"], color='r',alpha = 0.6, label='Loss')
plt.xlabel('Epochs')
plt.ylabel(r'VAE Loss')
plt.legend(bbox_to_anchor=(1.01, 1), loc=2)


#Encode rna data into latent space and save this 
# Model to compress input
encoder = Model(rnaseq_input, z_mean_encoded)
z = encoder.predict_on_batch(x_test)
z = pd.DataFrame(z, index=x_test.index)
z.columns = z.columns + 1


#Built a generator to sample from the latent space (the learned distribution) and decode back into mRNA form
# build a generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim, ))  # can generate from any sampled z vector
_x_decoded_mean = decoder_to_reconstruct(decoder_input)
decoder = Model(decoder_input, _x_decoded_mean)


#Encodes then decodes the data and compares these (input vs output) as confirmation that training has been successful. Plots 
# How well does the model reconstruct the input RNAseq data
x_reconstruct = decoder.predict(np.array(z))
x_reconstruct = pd.DataFrame(x_reconstruct, index=x_test.index,
                                        columns=x_test.columns)

reconstruction_fidelity = x_test - x_reconstruct
gene_mean = reconstruction_fidelity.mean(axis=0)
gene_abssum = reconstruction_fidelity.abs().sum(axis=0).divide(x_test.shape[0])
gene_summary = pd.DataFrame([gene_mean, gene_abssum], index=['Error mean', 'Error norm(abs(sum))']).T
gene_summary.sort_values(by='Error norm(abs(sum))', ascending=False).head()

# Mean of gene reconstruction vs. absolute reconstructed difference per sample
sns.jointplot('Error mean', 'Error norm(abs(sum))', data=gene_summary, stat_func=None);


#Compare these two. One is the true data, the other is the encoded then decoded data. 
print('Truth:')
print(x_test.head(5))
print('Reconstructed truth:')
print(x_reconstruct.head(5))





resistant_patients = np.load('./Data/Lists/resistant_patients.npy')
present_resistant_patients = np.intersect1d(resistant_patients,x_test.index.values)
sensitive_patients = np.load('./Data/Lists/sensitive_patients.npy')
present_sensitive_patients = np.intersect1d(sensitive_patients,x_test.index.values)

p_values = np.empty((latent_dim))
for i in range(latent_dim):
    latent_var_list = z.iloc[:,i]
    r = latent_var_list.reindex(present_resistant_patients).values
    s = latent_var_list.reindex(present_sensitive_patients).values
    p_values[i] = ks_2samp(r,s)[1]
adjusted_p_values = multipletests(p_values, method='fdr_bh', alpha=0.25)[1]
significant_list = (adjusted_p_values < 0.25)
significant_LVs = np.array([p_values,np.arange(latent_dim)]).T[np.array([p_values,np.arange(latent_dim)])[0].argsort()].T[1][:sum(significant_list)].astype(int)

if len(significant_LVs) == 0:
    print("None of the latent variables were found to be significant, please try again.")

else:
    plotdata = []
    for i in range(len(lvs2plot)):
        plotdata = np.append(plotdata,encoded_rnaseq_df.iloc[:,lvs2plot[i]].reindex(present_resistant_patients).values)
        plotdata = np.append(plotdata,encoded_rnaseq_df.iloc[:,lvs2plot[i]].reindex(present_sensitive_patients).values)
    plotvariables = []
    for i in range(len(lvs2plot)):
        plotvariables = np.append(plotvariables,[int(lvs2plot[i]+1)]*len(encoded_rnaseq_df.iloc[:,lvs2plot[i]].reindex(present_resistant_patients).values))
        plotvariables = np.append(plotvariables,[int(lvs2plot[i]+1)]*len(encoded_rnaseq_df.iloc[:,lvs2plot[i]].reindex(present_sensitive_patients).values))
    plotlabels = []
    for i in range(len(lvs2plot)):
        plotlabels = np.append(plotlabels,['Resistant']*len(encoded_rnaseq_df.iloc[:,lvs2plot[i]].reindex(present_resistant_patients).values))
        plotlabels = np.append(plotlabels,['Sensitive']*len(encoded_rnaseq_df.iloc[:,lvs2plot[i]].reindex(present_sensitive_patients).values))
    
    df = pd.DataFrame([plotdata,plotvariables,plotlabels]).T
    df.columns=['Activation','VAE Latent Dimension','Classification']
    df['Activation']=df['Activation'].astype(float)
    
    latent_variable_figure = plt.subplots()
    ax = sns.boxplot(x='VAE Latent Dimension', y='Activation', hue='Classification', data = df, palette=sns.color_palette(["#df5353", "#5fd35f"]))
    ax = sns.swarmplot(x='VAE Latent Dimension', y='Activation', hue='Classification', dodge=True, data = df, palette=sns.color_palette(["#d62728", "#2ca02c"]))
    handles, labels = ax.get_legend_handles_labels()
    l = plt.legend(handles[0:2], labels[0:2], loc=1, borderaxespad=0.4)
    
    LSP = -2*np.sum(np.log(np.delete(p_values,np.where(adjusted_p_values>0.25))))
    best_p = np.min(p_values)


    print('%g significant latent variables were found' )
    print('The latent space performance (LSP) was caluclated to be: %.3f' %LSP)
    print('The best latent variable (%g) seperates platinum resistant patients from sensitive patients with a p value of %.5f' %(np.argmin(p_values)+1,np.min(p_values)))        
    