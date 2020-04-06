import os
from tensorflow.python import pywrap_tensorflow
checkpoint_path = checkpoints_file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
var_dict = {
'rnn_model/RNN/MultiRNNCell/Cell0/LSTMCell/W_0':'rnn_model/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel' ,
'rnn_model/RNN/MultiRNNCell/Cell0/LSTMCell/B':'rnn_model/rnn/multi_rnn_cell/cell_0/lstm_cell/bias'
}

def rename(checkpoint_path, vardict, add_prefix, dry_run):
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    with tf.Session() as sess:
        for var_name in var_to_shape_map:
            # Load the variable
            var = reader.get_tensor(var_name)

            # Set the new name
            new_name = var_name
            if var_dict.get(var_name):
                new_name = new_name.replace(var_name, var_dict.get(var_name))
#             if add_prefix:
#                 new_name = add_prefix + new_name
#
            if dry_run:
                print('%s would be renamed to %s.' % (var_name, new_name))
            else:
                 print('Renaming %s to %s.' % (var_name, new_name))
                # Rename the variable
                 var = tf.Variable(var, name=new_name)
#
        if not dry_run:
            # Save the variables
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.save(sess, save_path=r'F:\magenta\test\\')
#
#
rename(checkpoint_path, vardict=var_dict, add_prefix=None, dry_run=None)
