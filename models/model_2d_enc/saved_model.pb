??	
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.02v2.3.0-rc2-23-gb36436b0878Ϫ
~
conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv1d_2/kernel
w
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*"
_output_shapes
:@*
dtype0
r
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_2/bias
k
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes
:@*
dtype0
~
conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ * 
shared_nameconv1d_3/kernel
w
#conv1d_3/kernel/Read/ReadVariableOpReadVariableOpconv1d_3/kernel*"
_output_shapes
:@ *
dtype0
r
conv1d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_3/bias
k
!conv1d_3/bias/Read/ReadVariableOpReadVariableOpconv1d_3/bias*
_output_shapes
: *
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	?@*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:@*
dtype0
v
z_mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namez_mean/kernel
o
!z_mean/kernel/Read/ReadVariableOpReadVariableOpz_mean/kernel*
_output_shapes

:@*
dtype0
n
z_mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namez_mean/bias
g
z_mean/bias/Read/ReadVariableOpReadVariableOpz_mean/bias*
_output_shapes
:*
dtype0
|
z_log_var/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_namez_log_var/kernel
u
$z_log_var/kernel/Read/ReadVariableOpReadVariableOpz_log_var/kernel*
_output_shapes

:@*
dtype0
t
z_log_var/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namez_log_var/bias
m
"z_log_var/bias/Read/ReadVariableOpReadVariableOpz_log_var/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?#
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?"
value?"B?" B?"
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
 trainable_variables
!regularization_losses
"	variables
#	keras_api
R
$trainable_variables
%regularization_losses
&	variables
'	keras_api
h

(kernel
)bias
*trainable_variables
+regularization_losses
,	variables
-	keras_api
h

.kernel
/bias
0trainable_variables
1regularization_losses
2	variables
3	keras_api
h

4kernel
5bias
6trainable_variables
7regularization_losses
8	variables
9	keras_api
R
:trainable_variables
;regularization_losses
<	variables
=	keras_api
F
0
1
2
3
(4
)5
.6
/7
48
59
 
F
0
1
2
3
(4
)5
.6
/7
48
59
?
trainable_variables
regularization_losses
	variables
>non_trainable_variables
?metrics
@layer_metrics

Alayers
Blayer_regularization_losses
 
[Y
VARIABLE_VALUEconv1d_2/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
trainable_variables
regularization_losses
	variables
Cnon_trainable_variables
Dmetrics
Elayer_metrics

Flayers
Glayer_regularization_losses
 
 
 
?
trainable_variables
regularization_losses
	variables
Hnon_trainable_variables
Imetrics
Jlayer_metrics

Klayers
Llayer_regularization_losses
[Y
VARIABLE_VALUEconv1d_3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
trainable_variables
regularization_losses
	variables
Mnon_trainable_variables
Nmetrics
Olayer_metrics

Players
Qlayer_regularization_losses
 
 
 
?
 trainable_variables
!regularization_losses
"	variables
Rnon_trainable_variables
Smetrics
Tlayer_metrics

Ulayers
Vlayer_regularization_losses
 
 
 
?
$trainable_variables
%regularization_losses
&	variables
Wnon_trainable_variables
Xmetrics
Ylayer_metrics

Zlayers
[layer_regularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1
?
*trainable_variables
+regularization_losses
,	variables
\non_trainable_variables
]metrics
^layer_metrics

_layers
`layer_regularization_losses
YW
VARIABLE_VALUEz_mean/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEz_mean/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1
 

.0
/1
?
0trainable_variables
1regularization_losses
2	variables
anon_trainable_variables
bmetrics
clayer_metrics

dlayers
elayer_regularization_losses
\Z
VARIABLE_VALUEz_log_var/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEz_log_var/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51
 

40
51
?
6trainable_variables
7regularization_losses
8	variables
fnon_trainable_variables
gmetrics
hlayer_metrics

ilayers
jlayer_regularization_losses
 
 
 
?
:trainable_variables
;regularization_losses
<	variables
knon_trainable_variables
lmetrics
mlayer_metrics

nlayers
olayer_regularization_losses
 
 
 
F
0
1
2
3
4
5
6
7
	8

9
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_input_3Placeholder*,
_output_shapes
:??????????*
dtype0*!
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3conv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/biasdense_2/kerneldense_2/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_2398161
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp#conv1d_3/kernel/Read/ReadVariableOp!conv1d_3/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp!z_mean/kernel/Read/ReadVariableOpz_mean/bias/Read/ReadVariableOp$z_log_var/kernel/Read/ReadVariableOp"z_log_var/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_2398585
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/biasdense_2/kerneldense_2/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_2398625??
?^
?
D__inference_encoder_layer_call_and_return_conditional_losses_2398321

inputs8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource8
4conv1d_3_conv1d_expanddims_1_readvariableop_resource,
(conv1d_3_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource)
%z_mean_matmul_readvariableop_resource*
&z_mean_biasadd_readvariableop_resource,
(z_log_var_matmul_readvariableop_resource-
)z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2??
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_2/conv1d/ExpandDims/dim?
conv1d_2/conv1d/ExpandDims
ExpandDimsinputs'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d_2/conv1d/ExpandDims?
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim?
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_2/conv1d/ExpandDims_1?
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????@*
paddingSAME*
strides
2
conv1d_2/conv1d?
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*,
_output_shapes
:??????????@*
squeeze_dims

?????????2
conv1d_2/conv1d/Squeeze?
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp?
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2
conv1d_2/BiasAddx
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
conv1d_2/Relu?
max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_2/ExpandDims/dim?
max_pooling1d_2/ExpandDims
ExpandDimsconv1d_2/Relu:activations:0'max_pooling1d_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????@2
max_pooling1d_2/ExpandDims?
max_pooling1d_2/MaxPoolMaxPool#max_pooling1d_2/ExpandDims:output:0*/
_output_shapes
:?????????C@*
ksize
*
paddingSAME*
strides
2
max_pooling1d_2/MaxPool?
max_pooling1d_2/SqueezeSqueeze max_pooling1d_2/MaxPool:output:0*
T0*+
_output_shapes
:?????????C@*
squeeze_dims
2
max_pooling1d_2/Squeeze?
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_3/conv1d/ExpandDims/dim?
conv1d_3/conv1d/ExpandDims
ExpandDims max_pooling1d_2/Squeeze:output:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????C@2
conv1d_3/conv1d/ExpandDims?
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dim?
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d_3/conv1d/ExpandDims_1?
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????C *
paddingSAME*
strides
2
conv1d_3/conv1d?
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*+
_output_shapes
:?????????C *
squeeze_dims

?????????2
conv1d_3/conv1d/Squeeze?
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_3/BiasAdd/ReadVariableOp?
conv1d_3/BiasAddBiasAdd conv1d_3/conv1d/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????C 2
conv1d_3/BiasAddw
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????C 2
conv1d_3/Relu?
max_pooling1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_3/ExpandDims/dim?
max_pooling1d_3/ExpandDims
ExpandDimsconv1d_3/Relu:activations:0'max_pooling1d_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????C 2
max_pooling1d_3/ExpandDims?
max_pooling1d_3/MaxPoolMaxPool#max_pooling1d_3/ExpandDims:output:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2
max_pooling1d_3/MaxPool?
max_pooling1d_3/SqueezeSqueeze max_pooling1d_3/MaxPool:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims
2
max_pooling1d_3/Squeezes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_1/Const?
flatten_1/ReshapeReshape max_pooling1d_3/Squeeze:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_1/Reshape?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_2/Relu?
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
z_mean/MatMul/ReadVariableOp?
z_mean/MatMulMatMuldense_2/Relu:activations:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_mean/MatMul?
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
z_mean/BiasAdd/ReadVariableOp?
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_mean/BiasAdd?
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02!
z_log_var/MatMul/ReadVariableOp?
z_log_var/MatMulMatMuldense_2/Relu:activations:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_log_var/MatMul?
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 z_log_var/BiasAdd/ReadVariableOp?
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_log_var/BiasAddk
sampling_1/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling_1/Shape?
sampling_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
sampling_1/strided_slice/stack?
 sampling_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling_1/strided_slice/stack_1?
 sampling_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling_1/strided_slice/stack_2?
sampling_1/strided_sliceStridedSlicesampling_1/Shape:output:0'sampling_1/strided_slice/stack:output:0)sampling_1/strided_slice/stack_1:output:0)sampling_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling_1/strided_sliceo
sampling_1/Shape_1Shapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling_1/Shape_1?
 sampling_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2"
 sampling_1/strided_slice_1/stack?
"sampling_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"sampling_1/strided_slice_1/stack_1?
"sampling_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"sampling_1/strided_slice_1/stack_2?
sampling_1/strided_slice_1StridedSlicesampling_1/Shape_1:output:0)sampling_1/strided_slice_1/stack:output:0+sampling_1/strided_slice_1/stack_1:output:0+sampling_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling_1/strided_slice_1?
sampling_1/random_normal/shapePack!sampling_1/strided_slice:output:0#sampling_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2 
sampling_1/random_normal/shape?
sampling_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sampling_1/random_normal/mean?
sampling_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
sampling_1/random_normal/stddev?
-sampling_1/random_normal/RandomStandardNormalRandomStandardNormal'sampling_1/random_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed2???2/
-sampling_1/random_normal/RandomStandardNormal?
sampling_1/random_normal/mulMul6sampling_1/random_normal/RandomStandardNormal:output:0(sampling_1/random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2
sampling_1/random_normal/mul?
sampling_1/random_normalAdd sampling_1/random_normal/mul:z:0&sampling_1/random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2
sampling_1/random_normali
sampling_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sampling_1/mul/x?
sampling_1/mulMulsampling_1/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sampling_1/mulm
sampling_1/ExpExpsampling_1/mul:z:0*
T0*'
_output_shapes
:?????????2
sampling_1/Exp?
sampling_1/mul_1Mulsampling_1/Exp:y:0sampling_1/random_normal:z:0*
T0*'
_output_shapes
:?????????2
sampling_1/mul_1?
sampling_1/addAddV2z_mean/BiasAdd:output:0sampling_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2
sampling_1/addk
IdentityIdentityz_mean/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identityr

Identity_1Identityz_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity_1j

Identity_2Identitysampling_1/add:z:0*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*S
_input_shapesB
@:??????????:::::::::::T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?-
?
#__inference__traced_restore_2398625
file_prefix$
 assignvariableop_conv1d_2_kernel$
 assignvariableop_1_conv1d_2_bias&
"assignvariableop_2_conv1d_3_kernel$
 assignvariableop_3_conv1d_3_bias%
!assignvariableop_4_dense_2_kernel#
assignvariableop_5_dense_2_bias$
 assignvariableop_6_z_mean_kernel"
assignvariableop_7_z_mean_bias'
#assignvariableop_8_z_log_var_kernel%
!assignvariableop_9_z_log_var_bias
identity_11??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_conv1d_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp assignvariableop_6_z_mean_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_z_mean_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_z_log_var_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_z_log_var_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10?
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_11"#
identity_11Identity_11:output:0*=
_input_shapes,
*: ::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?/
?
D__inference_encoder_layer_call_and_return_conditional_losses_2397966
input_3
conv1d_2_2397796
conv1d_2_2397798
conv1d_3_2397829
conv1d_3_2397831
dense_2_2397871
dense_2_2397873
z_mean_2397897
z_mean_2397899
z_log_var_2397923
z_log_var_2397925
identity

identity_1

identity_2?? conv1d_2/StatefulPartitionedCall? conv1d_3/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?"sampling_1/StatefulPartitionedCall?!z_log_var/StatefulPartitionedCall?z_mean/StatefulPartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallinput_3conv1d_2_2397796conv1d_2_2397798*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv1d_2_layer_call_and_return_conditional_losses_23977852"
 conv1d_2/StatefulPartitionedCall?
max_pooling1d_2/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????C@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_23977442!
max_pooling1d_2/PartitionedCall?
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_2/PartitionedCall:output:0conv1d_3_2397829conv1d_3_2397831*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????C *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv1d_3_layer_call_and_return_conditional_losses_23978182"
 conv1d_3/StatefulPartitionedCall?
max_pooling1d_3/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling1d_3_layer_call_and_return_conditional_losses_23977592!
max_pooling1d_3/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall(max_pooling1d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_23978412
flatten_1/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_2397871dense_2_2397873*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_23978602!
dense_2/StatefulPartitionedCall?
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0z_mean_2397897z_mean_2397899*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_z_mean_layer_call_and_return_conditional_losses_23978862 
z_mean/StatefulPartitionedCall?
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0z_log_var_2397923z_log_var_2397925*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_z_log_var_layer_call_and_return_conditional_losses_23979122#
!z_log_var/StatefulPartitionedCall?
"sampling_1/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sampling_1_layer_call_and_return_conditional_losses_23979542$
"sampling_1/StatefulPartitionedCall?
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity+sampling_1/StatefulPartitionedCall:output:0!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*S
_input_shapesB
@:??????????::::::::::2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2H
"sampling_1/StatefulPartitionedCall"sampling_1/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_3
?
?
)__inference_encoder_layer_call_fn_2398130
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_23981032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*S
_input_shapesB
@:??????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_3
?

*__inference_conv1d_2_layer_call_fn_2398404

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv1d_2_layer_call_and_return_conditional_losses_23977852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_encoder_layer_call_fn_2398379

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_23981032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*S
_input_shapesB
@:??????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv1d_3_layer_call_and_return_conditional_losses_2397818

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????C@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????C *
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????C *
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????C 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????C 2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:?????????C 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????C@:::S O
+
_output_shapes
:?????????C@
 
_user_specified_nameinputs
?
?
E__inference_conv1d_2_layer_call_and_return_conditional_losses_2398395

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????@*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????:::T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_encoder_layer_call_fn_2398066
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_23980392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*S
_input_shapesB
@:??????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_3
?
?
C__inference_z_mean_layer_call_and_return_conditional_losses_2397886

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_2398161
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_23977352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*S
_input_shapesB
@:??????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_3
?^
?
D__inference_encoder_layer_call_and_return_conditional_losses_2398241

inputs8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource8
4conv1d_3_conv1d_expanddims_1_readvariableop_resource,
(conv1d_3_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource)
%z_mean_matmul_readvariableop_resource*
&z_mean_biasadd_readvariableop_resource,
(z_log_var_matmul_readvariableop_resource-
)z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2??
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_2/conv1d/ExpandDims/dim?
conv1d_2/conv1d/ExpandDims
ExpandDimsinputs'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d_2/conv1d/ExpandDims?
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim?
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_2/conv1d/ExpandDims_1?
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????@*
paddingSAME*
strides
2
conv1d_2/conv1d?
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*,
_output_shapes
:??????????@*
squeeze_dims

?????????2
conv1d_2/conv1d/Squeeze?
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp?
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2
conv1d_2/BiasAddx
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
conv1d_2/Relu?
max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_2/ExpandDims/dim?
max_pooling1d_2/ExpandDims
ExpandDimsconv1d_2/Relu:activations:0'max_pooling1d_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????@2
max_pooling1d_2/ExpandDims?
max_pooling1d_2/MaxPoolMaxPool#max_pooling1d_2/ExpandDims:output:0*/
_output_shapes
:?????????C@*
ksize
*
paddingSAME*
strides
2
max_pooling1d_2/MaxPool?
max_pooling1d_2/SqueezeSqueeze max_pooling1d_2/MaxPool:output:0*
T0*+
_output_shapes
:?????????C@*
squeeze_dims
2
max_pooling1d_2/Squeeze?
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_3/conv1d/ExpandDims/dim?
conv1d_3/conv1d/ExpandDims
ExpandDims max_pooling1d_2/Squeeze:output:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????C@2
conv1d_3/conv1d/ExpandDims?
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dim?
conv1d_3/conv1d/ExpandDims_1
ExpandDims3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d_3/conv1d/ExpandDims_1?
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????C *
paddingSAME*
strides
2
conv1d_3/conv1d?
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*+
_output_shapes
:?????????C *
squeeze_dims

?????????2
conv1d_3/conv1d/Squeeze?
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_3/BiasAdd/ReadVariableOp?
conv1d_3/BiasAddBiasAdd conv1d_3/conv1d/Squeeze:output:0'conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????C 2
conv1d_3/BiasAddw
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????C 2
conv1d_3/Relu?
max_pooling1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_3/ExpandDims/dim?
max_pooling1d_3/ExpandDims
ExpandDimsconv1d_3/Relu:activations:0'max_pooling1d_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????C 2
max_pooling1d_3/ExpandDims?
max_pooling1d_3/MaxPoolMaxPool#max_pooling1d_3/ExpandDims:output:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2
max_pooling1d_3/MaxPool?
max_pooling1d_3/SqueezeSqueeze max_pooling1d_3/MaxPool:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims
2
max_pooling1d_3/Squeezes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_1/Const?
flatten_1/ReshapeReshape max_pooling1d_3/Squeeze:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_1/Reshape?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_2/Relu?
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
z_mean/MatMul/ReadVariableOp?
z_mean/MatMulMatMuldense_2/Relu:activations:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_mean/MatMul?
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
z_mean/BiasAdd/ReadVariableOp?
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_mean/BiasAdd?
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02!
z_log_var/MatMul/ReadVariableOp?
z_log_var/MatMulMatMuldense_2/Relu:activations:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_log_var/MatMul?
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 z_log_var/BiasAdd/ReadVariableOp?
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_log_var/BiasAddk
sampling_1/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling_1/Shape?
sampling_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
sampling_1/strided_slice/stack?
 sampling_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling_1/strided_slice/stack_1?
 sampling_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling_1/strided_slice/stack_2?
sampling_1/strided_sliceStridedSlicesampling_1/Shape:output:0'sampling_1/strided_slice/stack:output:0)sampling_1/strided_slice/stack_1:output:0)sampling_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling_1/strided_sliceo
sampling_1/Shape_1Shapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling_1/Shape_1?
 sampling_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2"
 sampling_1/strided_slice_1/stack?
"sampling_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"sampling_1/strided_slice_1/stack_1?
"sampling_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"sampling_1/strided_slice_1/stack_2?
sampling_1/strided_slice_1StridedSlicesampling_1/Shape_1:output:0)sampling_1/strided_slice_1/stack:output:0+sampling_1/strided_slice_1/stack_1:output:0+sampling_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling_1/strided_slice_1?
sampling_1/random_normal/shapePack!sampling_1/strided_slice:output:0#sampling_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2 
sampling_1/random_normal/shape?
sampling_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sampling_1/random_normal/mean?
sampling_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
sampling_1/random_normal/stddev?
-sampling_1/random_normal/RandomStandardNormalRandomStandardNormal'sampling_1/random_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed2???2/
-sampling_1/random_normal/RandomStandardNormal?
sampling_1/random_normal/mulMul6sampling_1/random_normal/RandomStandardNormal:output:0(sampling_1/random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2
sampling_1/random_normal/mul?
sampling_1/random_normalAdd sampling_1/random_normal/mul:z:0&sampling_1/random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2
sampling_1/random_normali
sampling_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sampling_1/mul/x?
sampling_1/mulMulsampling_1/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sampling_1/mulm
sampling_1/ExpExpsampling_1/mul:z:0*
T0*'
_output_shapes
:?????????2
sampling_1/Exp?
sampling_1/mul_1Mulsampling_1/Exp:y:0sampling_1/random_normal:z:0*
T0*'
_output_shapes
:?????????2
sampling_1/mul_1?
sampling_1/addAddV2z_mean/BiasAdd:output:0sampling_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2
sampling_1/addk
IdentityIdentityz_mean/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identityr

Identity_1Identityz_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity_1j

Identity_2Identitysampling_1/add:z:0*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*S
_input_shapesB
@:??????????:::::::::::T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?/
?
D__inference_encoder_layer_call_and_return_conditional_losses_2398039

inputs
conv1d_2_2398007
conv1d_2_2398009
conv1d_3_2398013
conv1d_3_2398015
dense_2_2398020
dense_2_2398022
z_mean_2398025
z_mean_2398027
z_log_var_2398030
z_log_var_2398032
identity

identity_1

identity_2?? conv1d_2/StatefulPartitionedCall? conv1d_3/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?"sampling_1/StatefulPartitionedCall?!z_log_var/StatefulPartitionedCall?z_mean/StatefulPartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_2_2398007conv1d_2_2398009*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv1d_2_layer_call_and_return_conditional_losses_23977852"
 conv1d_2/StatefulPartitionedCall?
max_pooling1d_2/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????C@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_23977442!
max_pooling1d_2/PartitionedCall?
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_2/PartitionedCall:output:0conv1d_3_2398013conv1d_3_2398015*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????C *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv1d_3_layer_call_and_return_conditional_losses_23978182"
 conv1d_3/StatefulPartitionedCall?
max_pooling1d_3/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling1d_3_layer_call_and_return_conditional_losses_23977592!
max_pooling1d_3/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall(max_pooling1d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_23978412
flatten_1/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_2398020dense_2_2398022*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_23978602!
dense_2/StatefulPartitionedCall?
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0z_mean_2398025z_mean_2398027*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_z_mean_layer_call_and_return_conditional_losses_23978862 
z_mean/StatefulPartitionedCall?
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0z_log_var_2398030z_log_var_2398032*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_z_log_var_layer_call_and_return_conditional_losses_23979122#
!z_log_var/StatefulPartitionedCall?
"sampling_1/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sampling_1_layer_call_and_return_conditional_losses_23979542$
"sampling_1/StatefulPartitionedCall?
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity+sampling_1/StatefulPartitionedCall:output:0!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*S
_input_shapesB
@:??????????::::::::::2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2H
"sampling_1/StatefulPartitionedCall"sampling_1/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_z_log_var_layer_call_and_return_conditional_losses_2397912

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_2398435

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
M
1__inference_max_pooling1d_3_layer_call_fn_2397765

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling1d_3_layer_call_and_return_conditional_losses_23977592
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?/
?
D__inference_encoder_layer_call_and_return_conditional_losses_2398103

inputs
conv1d_2_2398071
conv1d_2_2398073
conv1d_3_2398077
conv1d_3_2398079
dense_2_2398084
dense_2_2398086
z_mean_2398089
z_mean_2398091
z_log_var_2398094
z_log_var_2398096
identity

identity_1

identity_2?? conv1d_2/StatefulPartitionedCall? conv1d_3/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?"sampling_1/StatefulPartitionedCall?!z_log_var/StatefulPartitionedCall?z_mean/StatefulPartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_2_2398071conv1d_2_2398073*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv1d_2_layer_call_and_return_conditional_losses_23977852"
 conv1d_2/StatefulPartitionedCall?
max_pooling1d_2/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????C@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_23977442!
max_pooling1d_2/PartitionedCall?
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_2/PartitionedCall:output:0conv1d_3_2398077conv1d_3_2398079*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????C *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv1d_3_layer_call_and_return_conditional_losses_23978182"
 conv1d_3/StatefulPartitionedCall?
max_pooling1d_3/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling1d_3_layer_call_and_return_conditional_losses_23977592!
max_pooling1d_3/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall(max_pooling1d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_23978412
flatten_1/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_2398084dense_2_2398086*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_23978602!
dense_2/StatefulPartitionedCall?
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0z_mean_2398089z_mean_2398091*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_z_mean_layer_call_and_return_conditional_losses_23978862 
z_mean/StatefulPartitionedCall?
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0z_log_var_2398094z_log_var_2398096*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_z_log_var_layer_call_and_return_conditional_losses_23979122#
!z_log_var/StatefulPartitionedCall?
"sampling_1/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sampling_1_layer_call_and_return_conditional_losses_23979542$
"sampling_1/StatefulPartitionedCall?
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity+sampling_1/StatefulPartitionedCall:output:0!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*S
_input_shapesB
@:??????????::::::::::2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2H
"sampling_1/StatefulPartitionedCall"sampling_1/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
~
)__inference_dense_2_layer_call_fn_2398460

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_23978602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_z_log_var_layer_call_fn_2398498

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_z_log_var_layer_call_and_return_conditional_losses_23979122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
D__inference_dense_2_layer_call_and_return_conditional_losses_2398451

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_2397744

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?/
?
D__inference_encoder_layer_call_and_return_conditional_losses_2398001
input_3
conv1d_2_2397969
conv1d_2_2397971
conv1d_3_2397975
conv1d_3_2397977
dense_2_2397982
dense_2_2397984
z_mean_2397987
z_mean_2397989
z_log_var_2397992
z_log_var_2397994
identity

identity_1

identity_2?? conv1d_2/StatefulPartitionedCall? conv1d_3/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?"sampling_1/StatefulPartitionedCall?!z_log_var/StatefulPartitionedCall?z_mean/StatefulPartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallinput_3conv1d_2_2397969conv1d_2_2397971*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv1d_2_layer_call_and_return_conditional_losses_23977852"
 conv1d_2/StatefulPartitionedCall?
max_pooling1d_2/PartitionedCallPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????C@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_23977442!
max_pooling1d_2/PartitionedCall?
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_2/PartitionedCall:output:0conv1d_3_2397975conv1d_3_2397977*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????C *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv1d_3_layer_call_and_return_conditional_losses_23978182"
 conv1d_3/StatefulPartitionedCall?
max_pooling1d_3/PartitionedCallPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling1d_3_layer_call_and_return_conditional_losses_23977592!
max_pooling1d_3/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall(max_pooling1d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_23978412
flatten_1/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_2397982dense_2_2397984*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_23978602!
dense_2/StatefulPartitionedCall?
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0z_mean_2397987z_mean_2397989*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_z_mean_layer_call_and_return_conditional_losses_23978862 
z_mean/StatefulPartitionedCall?
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0z_log_var_2397992z_log_var_2397994*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_z_log_var_layer_call_and_return_conditional_losses_23979122#
!z_log_var/StatefulPartitionedCall?
"sampling_1/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sampling_1_layer_call_and_return_conditional_losses_23979542$
"sampling_1/StatefulPartitionedCall?
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity+sampling_1/StatefulPartitionedCall:output:0!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*S
_input_shapesB
@:??????????::::::::::2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2H
"sampling_1/StatefulPartitionedCall"sampling_1/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_3
?
?
E__inference_conv1d_2_layer_call_and_return_conditional_losses_2397785

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????@*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????:::T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling1d_3_layer_call_and_return_conditional_losses_2397759

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
}
(__inference_z_mean_layer_call_fn_2398479

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_z_mean_layer_call_and_return_conditional_losses_23978862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
C__inference_z_mean_layer_call_and_return_conditional_losses_2398470

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?k
?
"__inference__wrapped_model_2397735
input_3@
<encoder_conv1d_2_conv1d_expanddims_1_readvariableop_resource4
0encoder_conv1d_2_biasadd_readvariableop_resource@
<encoder_conv1d_3_conv1d_expanddims_1_readvariableop_resource4
0encoder_conv1d_3_biasadd_readvariableop_resource2
.encoder_dense_2_matmul_readvariableop_resource3
/encoder_dense_2_biasadd_readvariableop_resource1
-encoder_z_mean_matmul_readvariableop_resource2
.encoder_z_mean_biasadd_readvariableop_resource4
0encoder_z_log_var_matmul_readvariableop_resource5
1encoder_z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2??
&encoder/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&encoder/conv1d_2/conv1d/ExpandDims/dim?
"encoder/conv1d_2/conv1d/ExpandDims
ExpandDimsinput_3/encoder/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2$
"encoder/conv1d_2/conv1d/ExpandDims?
3encoder/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<encoder_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype025
3encoder/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
(encoder/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(encoder/conv1d_2/conv1d/ExpandDims_1/dim?
$encoder/conv1d_2/conv1d/ExpandDims_1
ExpandDims;encoder/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:01encoder/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2&
$encoder/conv1d_2/conv1d/ExpandDims_1?
encoder/conv1d_2/conv1dConv2D+encoder/conv1d_2/conv1d/ExpandDims:output:0-encoder/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????@*
paddingSAME*
strides
2
encoder/conv1d_2/conv1d?
encoder/conv1d_2/conv1d/SqueezeSqueeze encoder/conv1d_2/conv1d:output:0*
T0*,
_output_shapes
:??????????@*
squeeze_dims

?????????2!
encoder/conv1d_2/conv1d/Squeeze?
'encoder/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'encoder/conv1d_2/BiasAdd/ReadVariableOp?
encoder/conv1d_2/BiasAddBiasAdd(encoder/conv1d_2/conv1d/Squeeze:output:0/encoder/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2
encoder/conv1d_2/BiasAdd?
encoder/conv1d_2/ReluRelu!encoder/conv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
encoder/conv1d_2/Relu?
&encoder/max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&encoder/max_pooling1d_2/ExpandDims/dim?
"encoder/max_pooling1d_2/ExpandDims
ExpandDims#encoder/conv1d_2/Relu:activations:0/encoder/max_pooling1d_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????@2$
"encoder/max_pooling1d_2/ExpandDims?
encoder/max_pooling1d_2/MaxPoolMaxPool+encoder/max_pooling1d_2/ExpandDims:output:0*/
_output_shapes
:?????????C@*
ksize
*
paddingSAME*
strides
2!
encoder/max_pooling1d_2/MaxPool?
encoder/max_pooling1d_2/SqueezeSqueeze(encoder/max_pooling1d_2/MaxPool:output:0*
T0*+
_output_shapes
:?????????C@*
squeeze_dims
2!
encoder/max_pooling1d_2/Squeeze?
&encoder/conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&encoder/conv1d_3/conv1d/ExpandDims/dim?
"encoder/conv1d_3/conv1d/ExpandDims
ExpandDims(encoder/max_pooling1d_2/Squeeze:output:0/encoder/conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????C@2$
"encoder/conv1d_3/conv1d/ExpandDims?
3encoder/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<encoder_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype025
3encoder/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?
(encoder/conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(encoder/conv1d_3/conv1d/ExpandDims_1/dim?
$encoder/conv1d_3/conv1d/ExpandDims_1
ExpandDims;encoder/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:01encoder/conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2&
$encoder/conv1d_3/conv1d/ExpandDims_1?
encoder/conv1d_3/conv1dConv2D+encoder/conv1d_3/conv1d/ExpandDims:output:0-encoder/conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????C *
paddingSAME*
strides
2
encoder/conv1d_3/conv1d?
encoder/conv1d_3/conv1d/SqueezeSqueeze encoder/conv1d_3/conv1d:output:0*
T0*+
_output_shapes
:?????????C *
squeeze_dims

?????????2!
encoder/conv1d_3/conv1d/Squeeze?
'encoder/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'encoder/conv1d_3/BiasAdd/ReadVariableOp?
encoder/conv1d_3/BiasAddBiasAdd(encoder/conv1d_3/conv1d/Squeeze:output:0/encoder/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????C 2
encoder/conv1d_3/BiasAdd?
encoder/conv1d_3/ReluRelu!encoder/conv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????C 2
encoder/conv1d_3/Relu?
&encoder/max_pooling1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&encoder/max_pooling1d_3/ExpandDims/dim?
"encoder/max_pooling1d_3/ExpandDims
ExpandDims#encoder/conv1d_3/Relu:activations:0/encoder/max_pooling1d_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????C 2$
"encoder/max_pooling1d_3/ExpandDims?
encoder/max_pooling1d_3/MaxPoolMaxPool+encoder/max_pooling1d_3/ExpandDims:output:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2!
encoder/max_pooling1d_3/MaxPool?
encoder/max_pooling1d_3/SqueezeSqueeze(encoder/max_pooling1d_3/MaxPool:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims
2!
encoder/max_pooling1d_3/Squeeze?
encoder/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
encoder/flatten_1/Const?
encoder/flatten_1/ReshapeReshape(encoder/max_pooling1d_3/Squeeze:output:0 encoder/flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
encoder/flatten_1/Reshape?
%encoder/dense_2/MatMul/ReadVariableOpReadVariableOp.encoder_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02'
%encoder/dense_2/MatMul/ReadVariableOp?
encoder/dense_2/MatMulMatMul"encoder/flatten_1/Reshape:output:0-encoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
encoder/dense_2/MatMul?
&encoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&encoder/dense_2/BiasAdd/ReadVariableOp?
encoder/dense_2/BiasAddBiasAdd encoder/dense_2/MatMul:product:0.encoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
encoder/dense_2/BiasAdd?
encoder/dense_2/ReluRelu encoder/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
encoder/dense_2/Relu?
$encoder/z_mean/MatMul/ReadVariableOpReadVariableOp-encoder_z_mean_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02&
$encoder/z_mean/MatMul/ReadVariableOp?
encoder/z_mean/MatMulMatMul"encoder/dense_2/Relu:activations:0,encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/z_mean/MatMul?
%encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp.encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%encoder/z_mean/BiasAdd/ReadVariableOp?
encoder/z_mean/BiasAddBiasAddencoder/z_mean/MatMul:product:0-encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/z_mean/BiasAdd?
'encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp0encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02)
'encoder/z_log_var/MatMul/ReadVariableOp?
encoder/z_log_var/MatMulMatMul"encoder/dense_2/Relu:activations:0/encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/z_log_var/MatMul?
(encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp1encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(encoder/z_log_var/BiasAdd/ReadVariableOp?
encoder/z_log_var/BiasAddBiasAdd"encoder/z_log_var/MatMul:product:00encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/z_log_var/BiasAdd?
encoder/sampling_1/ShapeShapeencoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
encoder/sampling_1/Shape?
&encoder/sampling_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&encoder/sampling_1/strided_slice/stack?
(encoder/sampling_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(encoder/sampling_1/strided_slice/stack_1?
(encoder/sampling_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(encoder/sampling_1/strided_slice/stack_2?
 encoder/sampling_1/strided_sliceStridedSlice!encoder/sampling_1/Shape:output:0/encoder/sampling_1/strided_slice/stack:output:01encoder/sampling_1/strided_slice/stack_1:output:01encoder/sampling_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 encoder/sampling_1/strided_slice?
encoder/sampling_1/Shape_1Shapeencoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
encoder/sampling_1/Shape_1?
(encoder/sampling_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(encoder/sampling_1/strided_slice_1/stack?
*encoder/sampling_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*encoder/sampling_1/strided_slice_1/stack_1?
*encoder/sampling_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*encoder/sampling_1/strided_slice_1/stack_2?
"encoder/sampling_1/strided_slice_1StridedSlice#encoder/sampling_1/Shape_1:output:01encoder/sampling_1/strided_slice_1/stack:output:03encoder/sampling_1/strided_slice_1/stack_1:output:03encoder/sampling_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"encoder/sampling_1/strided_slice_1?
&encoder/sampling_1/random_normal/shapePack)encoder/sampling_1/strided_slice:output:0+encoder/sampling_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2(
&encoder/sampling_1/random_normal/shape?
%encoder/sampling_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%encoder/sampling_1/random_normal/mean?
'encoder/sampling_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'encoder/sampling_1/random_normal/stddev?
5encoder/sampling_1/random_normal/RandomStandardNormalRandomStandardNormal/encoder/sampling_1/random_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed2???27
5encoder/sampling_1/random_normal/RandomStandardNormal?
$encoder/sampling_1/random_normal/mulMul>encoder/sampling_1/random_normal/RandomStandardNormal:output:00encoder/sampling_1/random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2&
$encoder/sampling_1/random_normal/mul?
 encoder/sampling_1/random_normalAdd(encoder/sampling_1/random_normal/mul:z:0.encoder/sampling_1/random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2"
 encoder/sampling_1/random_normaly
encoder/sampling_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
encoder/sampling_1/mul/x?
encoder/sampling_1/mulMul!encoder/sampling_1/mul/x:output:0"encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
encoder/sampling_1/mul?
encoder/sampling_1/ExpExpencoder/sampling_1/mul:z:0*
T0*'
_output_shapes
:?????????2
encoder/sampling_1/Exp?
encoder/sampling_1/mul_1Mulencoder/sampling_1/Exp:y:0$encoder/sampling_1/random_normal:z:0*
T0*'
_output_shapes
:?????????2
encoder/sampling_1/mul_1?
encoder/sampling_1/addAddV2encoder/z_mean/BiasAdd:output:0encoder/sampling_1/mul_1:z:0*
T0*'
_output_shapes
:?????????2
encoder/sampling_1/addn
IdentityIdentityencoder/sampling_1/add:z:0*
T0*'
_output_shapes
:?????????2

Identityz

Identity_1Identity"encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity_1w

Identity_2Identityencoder/z_mean/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*S
_input_shapesB
@:??????????:::::::::::U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_3
?
M
1__inference_max_pooling1d_2_layer_call_fn_2397750

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_23977442
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_encoder_layer_call_fn_2398350

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_23980392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*S
_input_shapesB
@:??????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_dense_2_layer_call_and_return_conditional_losses_2397860

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

*__inference_conv1d_3_layer_call_fn_2398429

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????C *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv1d_3_layer_call_and_return_conditional_losses_23978182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????C 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????C@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????C@
 
_user_specified_nameinputs
?
G
+__inference_flatten_1_layer_call_fn_2398440

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_23978412
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
u
,__inference_sampling_1_layer_call_fn_2398530
inputs_0
inputs_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sampling_1_layer_call_and_return_conditional_losses_23979542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
t
G__inference_sampling_1_layer_call_and_return_conditional_losses_2397954

inputs
inputs_1
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceH
Shape_1Shapeinputs*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2
random_normalS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x]
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:?????????2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:?????????2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????2
mul_1X
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
v
G__inference_sampling_1_layer_call_and_return_conditional_losses_2398524
inputs_0
inputs_1
identity?F
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceJ
Shape_1Shapeinputs_0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2
random_normalS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x]
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:?????????2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:?????????2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????2
mul_1Z
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?"
?
 __inference__traced_save_2398585
file_prefix.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop.
*savev2_conv1d_3_kernel_read_readvariableop,
(savev2_conv1d_3_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop,
(savev2_z_mean_kernel_read_readvariableop*
&savev2_z_mean_bias_read_readvariableop/
+savev2_z_log_var_kernel_read_readvariableop-
)savev2_z_log_var_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_bf9d7f1c88444da4a20a3451c54504ca/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop*savev2_conv1d_3_kernel_read_readvariableop(savev2_conv1d_3_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop(savev2_z_mean_kernel_read_readvariableop&savev2_z_mean_bias_read_readvariableop+savev2_z_log_var_kernel_read_readvariableop)savev2_z_log_var_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*p
_input_shapes_
]: :@:@:@ : :	?@:@:@::@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:@: 

_output_shapes
:@:($
"
_output_shapes
:@ : 

_output_shapes
: :%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::$	 

_output_shapes

:@: 


_output_shapes
::

_output_shapes
: 
?
?
F__inference_z_log_var_layer_call_and_return_conditional_losses_2398489

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_2397841

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
E__inference_conv1d_3_layer_call_and_return_conditional_losses_2398420

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity?y
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????C@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????C *
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????C *
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????C 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????C 2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:?????????C 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????C@:::S O
+
_output_shapes
:?????????C@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
@
input_35
serving_default_input_3:0??????????>

sampling_10
StatefulPartitionedCall:0?????????=
	z_log_var0
StatefulPartitionedCall:1?????????:
z_mean0
StatefulPartitionedCall:2?????????tensorflow/serving/predict:??
?K
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
trainable_variables
regularization_losses
	variables
	keras_api

signatures
*p&call_and_return_all_conditional_losses
q__call__
r_default_save_signature"?G
_tf_keras_network?G{"class_name": "Functional", "name": "encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 201, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "same", "data_format": "channels_last"}, "name": "max_pooling1d_2", "inbound_nodes": [[["conv1d_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_3", "inbound_nodes": [[["max_pooling1d_2", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "same", "data_format": "channels_last"}, "name": "max_pooling1d_3", "inbound_nodes": [[["conv1d_3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["max_pooling1d_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_var", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "sampling_1", "trainable": true, "dtype": "float32"}, "name": "sampling_1", "inbound_nodes": [[["z_mean", 0, 0, {}], ["z_log_var", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["z_mean", 0, 0], ["z_log_var", 0, 0], ["sampling_1", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 201, 6]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 201, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "same", "data_format": "channels_last"}, "name": "max_pooling1d_2", "inbound_nodes": [[["conv1d_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_3", "inbound_nodes": [[["max_pooling1d_2", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "same", "data_format": "channels_last"}, "name": "max_pooling1d_3", "inbound_nodes": [[["conv1d_3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["max_pooling1d_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_var", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "sampling_1", "trainable": true, "dtype": "float32"}, "name": "sampling_1", "inbound_nodes": [[["z_mean", 0, 0, {}], ["z_log_var", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["z_mean", 0, 0], ["z_log_var", 0, 0], ["sampling_1", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 201, 6]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 201, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
?	

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
s__call__
*t&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 201, 6]}}
?
trainable_variables
regularization_losses
	variables
	keras_api
u__call__
*v&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling1D", "name": "max_pooling1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "same", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
w__call__
*x&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 67, 64]}}
?
 trainable_variables
!regularization_losses
"	variables
#	keras_api
y__call__
*z&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling1D", "name": "max_pooling1d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "same", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
$trainable_variables
%regularization_losses
&	variables
'	keras_api
{__call__
*|&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

(kernel
)bias
*trainable_variables
+regularization_losses
,	variables
-	keras_api
}__call__
*~&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 736}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 736]}}
?

.kernel
/bias
0trainable_variables
1regularization_losses
2	variables
3	keras_api
__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "z_mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

4kernel
5bias
6trainable_variables
7regularization_losses
8	variables
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "z_log_var", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
:trainable_variables
;regularization_losses
<	variables
=	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Sampling", "name": "sampling_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "sampling_1", "trainable": true, "dtype": "float32"}}
f
0
1
2
3
(4
)5
.6
/7
48
59"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
(4
)5
.6
/7
48
59"
trackable_list_wrapper
?
trainable_variables
regularization_losses
	variables
>non_trainable_variables
?metrics
@layer_metrics

Alayers
Blayer_regularization_losses
q__call__
r_default_save_signature
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
%:#@2conv1d_2/kernel
:@2conv1d_2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
trainable_variables
regularization_losses
	variables
Cnon_trainable_variables
Dmetrics
Elayer_metrics

Flayers
Glayer_regularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
regularization_losses
	variables
Hnon_trainable_variables
Imetrics
Jlayer_metrics

Klayers
Llayer_regularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
%:#@ 2conv1d_3/kernel
: 2conv1d_3/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
trainable_variables
regularization_losses
	variables
Mnon_trainable_variables
Nmetrics
Olayer_metrics

Players
Qlayer_regularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 trainable_variables
!regularization_losses
"	variables
Rnon_trainable_variables
Smetrics
Tlayer_metrics

Ulayers
Vlayer_regularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
$trainable_variables
%regularization_losses
&	variables
Wnon_trainable_variables
Xmetrics
Ylayer_metrics

Zlayers
[layer_regularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
!:	?@2dense_2/kernel
:@2dense_2/bias
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?
*trainable_variables
+regularization_losses
,	variables
\non_trainable_variables
]metrics
^layer_metrics

_layers
`layer_regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
:@2z_mean/kernel
:2z_mean/bias
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
0trainable_variables
1regularization_losses
2	variables
anon_trainable_variables
bmetrics
clayer_metrics

dlayers
elayer_regularization_losses
__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": @2z_log_var/kernel
:2z_log_var/bias
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
?
6trainable_variables
7regularization_losses
8	variables
fnon_trainable_variables
gmetrics
hlayer_metrics

ilayers
jlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
:trainable_variables
;regularization_losses
<	variables
knon_trainable_variables
lmetrics
mlayer_metrics

nlayers
olayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
D__inference_encoder_layer_call_and_return_conditional_losses_2398241
D__inference_encoder_layer_call_and_return_conditional_losses_2398001
D__inference_encoder_layer_call_and_return_conditional_losses_2398321
D__inference_encoder_layer_call_and_return_conditional_losses_2397966?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_encoder_layer_call_fn_2398350
)__inference_encoder_layer_call_fn_2398066
)__inference_encoder_layer_call_fn_2398130
)__inference_encoder_layer_call_fn_2398379?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
"__inference__wrapped_model_2397735?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *+?(
&?#
input_3??????????
?2?
*__inference_conv1d_2_layer_call_fn_2398404?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv1d_2_layer_call_and_return_conditional_losses_2398395?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_max_pooling1d_2_layer_call_fn_2397750?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
L__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_2397744?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
*__inference_conv1d_3_layer_call_fn_2398429?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv1d_3_layer_call_and_return_conditional_losses_2398420?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_max_pooling1d_3_layer_call_fn_2397765?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
L__inference_max_pooling1d_3_layer_call_and_return_conditional_losses_2397759?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
+__inference_flatten_1_layer_call_fn_2398440?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_flatten_1_layer_call_and_return_conditional_losses_2398435?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_2_layer_call_fn_2398460?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_2_layer_call_and_return_conditional_losses_2398451?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_z_mean_layer_call_fn_2398479?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_z_mean_layer_call_and_return_conditional_losses_2398470?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_z_log_var_layer_call_fn_2398498?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_z_log_var_layer_call_and_return_conditional_losses_2398489?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_sampling_1_layer_call_fn_2398530?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_sampling_1_layer_call_and_return_conditional_losses_2398524?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
4B2
%__inference_signature_wrapper_2398161input_3?
"__inference__wrapped_model_2397735?
()./455?2
+?(
&?#
input_3??????????
? "???
2

sampling_1$?!

sampling_1?????????
0
	z_log_var#? 
	z_log_var?????????
*
z_mean ?
z_mean??????????
E__inference_conv1d_2_layer_call_and_return_conditional_losses_2398395f4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????@
? ?
*__inference_conv1d_2_layer_call_fn_2398404Y4?1
*?'
%?"
inputs??????????
? "???????????@?
E__inference_conv1d_3_layer_call_and_return_conditional_losses_2398420d3?0
)?&
$?!
inputs?????????C@
? ")?&
?
0?????????C 
? ?
*__inference_conv1d_3_layer_call_fn_2398429W3?0
)?&
$?!
inputs?????????C@
? "??????????C ?
D__inference_dense_2_layer_call_and_return_conditional_losses_2398451]()0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? }
)__inference_dense_2_layer_call_fn_2398460P()0?-
&?#
!?
inputs??????????
? "??????????@?
D__inference_encoder_layer_call_and_return_conditional_losses_2397966?
()./45=?:
3?0
&?#
input_3??????????
p

 
? "j?g
`?]
?
0/0?????????
?
0/1?????????
?
0/2?????????
? ?
D__inference_encoder_layer_call_and_return_conditional_losses_2398001?
()./45=?:
3?0
&?#
input_3??????????
p 

 
? "j?g
`?]
?
0/0?????????
?
0/1?????????
?
0/2?????????
? ?
D__inference_encoder_layer_call_and_return_conditional_losses_2398241?
()./45<?9
2?/
%?"
inputs??????????
p

 
? "j?g
`?]
?
0/0?????????
?
0/1?????????
?
0/2?????????
? ?
D__inference_encoder_layer_call_and_return_conditional_losses_2398321?
()./45<?9
2?/
%?"
inputs??????????
p 

 
? "j?g
`?]
?
0/0?????????
?
0/1?????????
?
0/2?????????
? ?
)__inference_encoder_layer_call_fn_2398066?
()./45=?:
3?0
&?#
input_3??????????
p

 
? "Z?W
?
0?????????
?
1?????????
?
2??????????
)__inference_encoder_layer_call_fn_2398130?
()./45=?:
3?0
&?#
input_3??????????
p 

 
? "Z?W
?
0?????????
?
1?????????
?
2??????????
)__inference_encoder_layer_call_fn_2398350?
()./45<?9
2?/
%?"
inputs??????????
p

 
? "Z?W
?
0?????????
?
1?????????
?
2??????????
)__inference_encoder_layer_call_fn_2398379?
()./45<?9
2?/
%?"
inputs??????????
p 

 
? "Z?W
?
0?????????
?
1?????????
?
2??????????
F__inference_flatten_1_layer_call_and_return_conditional_losses_2398435]3?0
)?&
$?!
inputs????????? 
? "&?#
?
0??????????
? 
+__inference_flatten_1_layer_call_fn_2398440P3?0
)?&
$?!
inputs????????? 
? "????????????
L__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_2397744?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
1__inference_max_pooling1d_2_layer_call_fn_2397750wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
L__inference_max_pooling1d_3_layer_call_and_return_conditional_losses_2397759?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
1__inference_max_pooling1d_3_layer_call_fn_2397765wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
G__inference_sampling_1_layer_call_and_return_conditional_losses_2398524?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
,__inference_sampling_1_layer_call_fn_2398530vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
%__inference_signature_wrapper_2398161?
()./45@?=
? 
6?3
1
input_3&?#
input_3??????????"???
2

sampling_1$?!

sampling_1?????????
0
	z_log_var#? 
	z_log_var?????????
*
z_mean ?
z_mean??????????
F__inference_z_log_var_layer_call_and_return_conditional_losses_2398489\45/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? ~
+__inference_z_log_var_layer_call_fn_2398498O45/?,
%?"
 ?
inputs?????????@
? "???????????
C__inference_z_mean_layer_call_and_return_conditional_losses_2398470\.//?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? {
(__inference_z_mean_layer_call_fn_2398479O.//?,
%?"
 ?
inputs?????????@
? "??????????