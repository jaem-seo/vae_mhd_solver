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
 ?"serve*2.3.02v2.3.0-rc2-23-gb36436b0878??
~
conv1d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv1d_6/kernel
w
#conv1d_6/kernel/Read/ReadVariableOpReadVariableOpconv1d_6/kernel*"
_output_shapes
:@*
dtype0
r
conv1d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_6/bias
k
!conv1d_6/bias/Read/ReadVariableOpReadVariableOpconv1d_6/bias*
_output_shapes
:@*
dtype0
~
conv1d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ * 
shared_nameconv1d_7/kernel
w
#conv1d_7/kernel/Read/ReadVariableOpReadVariableOpconv1d_7/kernel*"
_output_shapes
:@ *
dtype0
r
conv1d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_7/bias
k
!conv1d_7/bias/Read/ReadVariableOpReadVariableOpconv1d_7/bias*
_output_shapes
: *
dtype0
y
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*
shared_namedense_6/kernel
r
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes
:	?@*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
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
?&
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?%
value?%B?% B?%
?
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
#_self_saveable_object_factories

signatures
regularization_losses
trainable_variables
	variables
	keras_api
%
#_self_saveable_object_factories
?

kernel
bias
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
w
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
?

kernel
bias
# _self_saveable_object_factories
!regularization_losses
"trainable_variables
#	variables
$	keras_api
w
#%_self_saveable_object_factories
&regularization_losses
'trainable_variables
(	variables
)	keras_api
w
#*_self_saveable_object_factories
+regularization_losses
,trainable_variables
-	variables
.	keras_api
?

/kernel
0bias
#1_self_saveable_object_factories
2regularization_losses
3trainable_variables
4	variables
5	keras_api
?

6kernel
7bias
#8_self_saveable_object_factories
9regularization_losses
:trainable_variables
;	variables
<	keras_api
?

=kernel
>bias
#?_self_saveable_object_factories
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
w
#D_self_saveable_object_factories
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
 
 
 
F
0
1
2
3
/4
05
66
77
=8
>9
F
0
1
2
3
/4
05
66
77
=8
>9
?
Imetrics

Jlayers
regularization_losses
trainable_variables
	variables
Klayer_regularization_losses
Lnon_trainable_variables
Mlayer_metrics
 
[Y
VARIABLE_VALUEconv1d_6/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_6/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
?
Nmetrics

Olayers
regularization_losses
trainable_variables
	variables
Player_regularization_losses
Qnon_trainable_variables
Rlayer_metrics
 
 
 
 
?
Smetrics

Tlayers
regularization_losses
trainable_variables
	variables
Ulayer_regularization_losses
Vnon_trainable_variables
Wlayer_metrics
[Y
VARIABLE_VALUEconv1d_7/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_7/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
?
Xmetrics

Ylayers
!regularization_losses
"trainable_variables
#	variables
Zlayer_regularization_losses
[non_trainable_variables
\layer_metrics
 
 
 
 
?
]metrics

^layers
&regularization_losses
'trainable_variables
(	variables
_layer_regularization_losses
`non_trainable_variables
alayer_metrics
 
 
 
 
?
bmetrics

clayers
+regularization_losses
,trainable_variables
-	variables
dlayer_regularization_losses
enon_trainable_variables
flayer_metrics
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

/0
01

/0
01
?
gmetrics

hlayers
2regularization_losses
3trainable_variables
4	variables
ilayer_regularization_losses
jnon_trainable_variables
klayer_metrics
YW
VARIABLE_VALUEz_mean/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEz_mean/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

60
71

60
71
?
lmetrics

mlayers
9regularization_losses
:trainable_variables
;	variables
nlayer_regularization_losses
onon_trainable_variables
player_metrics
\Z
VARIABLE_VALUEz_log_var/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEz_log_var/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

=0
>1

=0
>1
?
qmetrics

rlayers
@regularization_losses
Atrainable_variables
B	variables
slayer_regularization_losses
tnon_trainable_variables
ulayer_metrics
 
 
 
 
?
vmetrics

wlayers
Eregularization_losses
Ftrainable_variables
G	variables
xlayer_regularization_losses
ynon_trainable_variables
zlayer_metrics
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
 
 
?
serving_default_input_7Placeholder*,
_output_shapes
:??????????*
dtype0*!
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_7conv1d_6/kernelconv1d_6/biasconv1d_7/kernelconv1d_7/biasdense_6/kerneldense_6/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/bias*
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
%__inference_signature_wrapper_5494783
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv1d_6/kernel/Read/ReadVariableOp!conv1d_6/bias/Read/ReadVariableOp#conv1d_7/kernel/Read/ReadVariableOp!conv1d_7/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp!z_mean/kernel/Read/ReadVariableOpz_mean/bias/Read/ReadVariableOp$z_log_var/kernel/Read/ReadVariableOp"z_log_var/bias/Read/ReadVariableOpConst*
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
 __inference__traced_save_5495135
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_6/kernelconv1d_6/biasconv1d_7/kernelconv1d_7/biasdense_6/kerneldense_6/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/bias*
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
#__inference__traced_restore_5495175??
?
h
L__inference_max_pooling1d_7_layer_call_and_return_conditional_losses_5494415

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
D__inference_encoder_layer_call_and_return_conditional_losses_5494725

inputs
conv1d_6_5494693
conv1d_6_5494695
conv1d_7_5494699
conv1d_7_5494701
dense_6_5494706
dense_6_5494708
z_mean_5494711
z_mean_5494713
z_log_var_5494716
z_log_var_5494718
identity

identity_1

identity_2?? conv1d_6/StatefulPartitionedCall? conv1d_7/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?"sampling_3/StatefulPartitionedCall?!z_log_var/StatefulPartitionedCall?z_mean/StatefulPartitionedCall?
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_6_5494693conv1d_6_5494695*
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
E__inference_conv1d_6_layer_call_and_return_conditional_losses_54944412"
 conv1d_6/StatefulPartitionedCall?
max_pooling1d_6/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling1d_6_layer_call_and_return_conditional_losses_54944002!
max_pooling1d_6/PartitionedCall?
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_6/PartitionedCall:output:0conv1d_7_5494699conv1d_7_5494701*
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
E__inference_conv1d_7_layer_call_and_return_conditional_losses_54944742"
 conv1d_7/StatefulPartitionedCall?
max_pooling1d_7/PartitionedCallPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling1d_7_layer_call_and_return_conditional_losses_54944152!
max_pooling1d_7/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall(max_pooling1d_7/PartitionedCall:output:0*
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
F__inference_flatten_3_layer_call_and_return_conditional_losses_54944972
flatten_3/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_5494706dense_6_5494708*
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
D__inference_dense_6_layer_call_and_return_conditional_losses_54945162!
dense_6/StatefulPartitionedCall?
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0z_mean_5494711z_mean_5494713*
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
C__inference_z_mean_layer_call_and_return_conditional_losses_54945422 
z_mean/StatefulPartitionedCall?
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0z_log_var_5494716z_log_var_5494718*
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
F__inference_z_log_var_layer_call_and_return_conditional_losses_54945682#
!z_log_var/StatefulPartitionedCall?
"sampling_3/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *3
f.R,
*__inference_restored_function_body_36627712$
"sampling_3/StatefulPartitionedCall?
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall#^sampling_3/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall#^sampling_3/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity+sampling_3/StatefulPartitionedCall:output:0!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall#^sampling_3/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*S
_input_shapesB
@:??????????::::::::::2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2H
"sampling_3/StatefulPartitionedCall"sampling_3/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

*__inference_conv1d_6_layer_call_fn_5494986

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
E__inference_conv1d_6_layer_call_and_return_conditional_losses_54944412
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
?/
?
D__inference_encoder_layer_call_and_return_conditional_losses_5494623
input_7
conv1d_6_5494591
conv1d_6_5494593
conv1d_7_5494597
conv1d_7_5494599
dense_6_5494604
dense_6_5494606
z_mean_5494609
z_mean_5494611
z_log_var_5494614
z_log_var_5494616
identity

identity_1

identity_2?? conv1d_6/StatefulPartitionedCall? conv1d_7/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?"sampling_3/StatefulPartitionedCall?!z_log_var/StatefulPartitionedCall?z_mean/StatefulPartitionedCall?
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCallinput_7conv1d_6_5494591conv1d_6_5494593*
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
E__inference_conv1d_6_layer_call_and_return_conditional_losses_54944412"
 conv1d_6/StatefulPartitionedCall?
max_pooling1d_6/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling1d_6_layer_call_and_return_conditional_losses_54944002!
max_pooling1d_6/PartitionedCall?
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_6/PartitionedCall:output:0conv1d_7_5494597conv1d_7_5494599*
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
E__inference_conv1d_7_layer_call_and_return_conditional_losses_54944742"
 conv1d_7/StatefulPartitionedCall?
max_pooling1d_7/PartitionedCallPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling1d_7_layer_call_and_return_conditional_losses_54944152!
max_pooling1d_7/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall(max_pooling1d_7/PartitionedCall:output:0*
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
F__inference_flatten_3_layer_call_and_return_conditional_losses_54944972
flatten_3/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_5494604dense_6_5494606*
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
D__inference_dense_6_layer_call_and_return_conditional_losses_54945162!
dense_6/StatefulPartitionedCall?
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0z_mean_5494609z_mean_5494611*
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
C__inference_z_mean_layer_call_and_return_conditional_losses_54945422 
z_mean/StatefulPartitionedCall?
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0z_log_var_5494614z_log_var_5494616*
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
F__inference_z_log_var_layer_call_and_return_conditional_losses_54945682#
!z_log_var/StatefulPartitionedCall?
"sampling_3/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *3
f.R,
*__inference_restored_function_body_36627712$
"sampling_3/StatefulPartitionedCall?
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall#^sampling_3/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall#^sampling_3/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity+sampling_3/StatefulPartitionedCall:output:0!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall#^sampling_3/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*S
_input_shapesB
@:??????????::::::::::2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2H
"sampling_3/StatefulPartitionedCall"sampling_3/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_7
?
b
F__inference_flatten_3_layer_call_and_return_conditional_losses_5495017

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
?
~
)__inference_dense_6_layer_call_fn_5495042

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
D__inference_dense_6_layer_call_and_return_conditional_losses_54945162
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
?
G
+__inference_flatten_3_layer_call_fn_5495022

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
F__inference_flatten_3_layer_call_and_return_conditional_losses_54944972
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
?/
?
D__inference_encoder_layer_call_and_return_conditional_losses_5494588
input_7
conv1d_6_5494452
conv1d_6_5494454
conv1d_7_5494485
conv1d_7_5494487
dense_6_5494527
dense_6_5494529
z_mean_5494553
z_mean_5494555
z_log_var_5494579
z_log_var_5494581
identity

identity_1

identity_2?? conv1d_6/StatefulPartitionedCall? conv1d_7/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?"sampling_3/StatefulPartitionedCall?!z_log_var/StatefulPartitionedCall?z_mean/StatefulPartitionedCall?
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCallinput_7conv1d_6_5494452conv1d_6_5494454*
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
E__inference_conv1d_6_layer_call_and_return_conditional_losses_54944412"
 conv1d_6/StatefulPartitionedCall?
max_pooling1d_6/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling1d_6_layer_call_and_return_conditional_losses_54944002!
max_pooling1d_6/PartitionedCall?
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_6/PartitionedCall:output:0conv1d_7_5494485conv1d_7_5494487*
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
E__inference_conv1d_7_layer_call_and_return_conditional_losses_54944742"
 conv1d_7/StatefulPartitionedCall?
max_pooling1d_7/PartitionedCallPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling1d_7_layer_call_and_return_conditional_losses_54944152!
max_pooling1d_7/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall(max_pooling1d_7/PartitionedCall:output:0*
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
F__inference_flatten_3_layer_call_and_return_conditional_losses_54944972
flatten_3/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_5494527dense_6_5494529*
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
D__inference_dense_6_layer_call_and_return_conditional_losses_54945162!
dense_6/StatefulPartitionedCall?
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0z_mean_5494553z_mean_5494555*
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
C__inference_z_mean_layer_call_and_return_conditional_losses_54945422 
z_mean/StatefulPartitionedCall?
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0z_log_var_5494579z_log_var_5494581*
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
F__inference_z_log_var_layer_call_and_return_conditional_losses_54945682#
!z_log_var/StatefulPartitionedCall?
"sampling_3/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *3
f.R,
*__inference_restored_function_body_36627712$
"sampling_3/StatefulPartitionedCall?
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall#^sampling_3/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall#^sampling_3/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity+sampling_3/StatefulPartitionedCall:output:0!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall#^sampling_3/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*S
_input_shapesB
@:??????????::::::::::2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2H
"sampling_3/StatefulPartitionedCall"sampling_3/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_7
?
?
E__forward_sampling_3_layer_call_and_return_conditional_losses_3664577

inputs_0_0

inputs_1_0
identity
inputs_0	
mul_1
exp
random_normal	
mul_x
inputs_1?H
ShapeShape
inputs_0_0*
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
strided_sliceL
Shape_1Shape
inputs_0_0*
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
dtype0*
seed2??2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2
random_normal/muld
random_normal_0Addrandom_normal/mul:z:0random_normal/mean:output:0*
T02
random_normalS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x_
mulMulmul/x:output:0
inputs_1_0*
T0*'
_output_shapes
:?????????2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:?????????2
Exp>
mul_1_0MulExp:y:0random_normal_0:z:0*
T02
mul_1^
addAddV2
inputs_0_0mul_1_0:z:0*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
expExp:y:0"
identityIdentity:output:0"
inputs_0
inputs_0_0"
inputs_1
inputs_1_0"
mul_1mul_1_0:z:0"
mul_xmul/x:output:0"$
random_normalrandom_normal_0:z:0*9
_input_shapes(
&:?????????:?????????*v
backward_function_name\Z__inference___backward_sampling_3_layer_call_and_return_conditional_losses_3664536_3664578:Q M
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
?
?
)__inference_encoder_layer_call_fn_5494932

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
D__inference_encoder_layer_call_and_return_conditional_losses_54946612
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
?
v
G__inference_sampling_3_layer_call_and_return_conditional_losses_3661957
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
dtype0*
seed2??2$
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
?
}
(__inference_z_mean_layer_call_fn_5495061

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
C__inference_z_mean_layer_call_and_return_conditional_losses_54945422
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
?"
?
 __inference__traced_save_5495135
file_prefix.
*savev2_conv1d_6_kernel_read_readvariableop,
(savev2_conv1d_6_bias_read_readvariableop.
*savev2_conv1d_7_kernel_read_readvariableop,
(savev2_conv1d_7_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop,
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
value3B1 B+_temp_f64568ccfba846858a84855c1d98049f/part2	
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1d_6_kernel_read_readvariableop(savev2_conv1d_6_bias_read_readvariableop*savev2_conv1d_7_kernel_read_readvariableop(savev2_conv1d_7_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop(savev2_z_mean_kernel_read_readvariableop&savev2_z_mean_bias_read_readvariableop+savev2_z_log_var_kernel_read_readvariableop)savev2_z_log_var_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
?
?
%__inference_signature_wrapper_5494783
input_7
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
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
"__inference__wrapped_model_54943912
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
_user_specified_name	input_7
?
?
)__inference_encoder_layer_call_fn_5494688
input_7
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
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
D__inference_encoder_layer_call_and_return_conditional_losses_54946612
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
_user_specified_name	input_7
?R
?
"__inference__wrapped_model_5494391
input_7@
<encoder_conv1d_6_conv1d_expanddims_1_readvariableop_resource4
0encoder_conv1d_6_biasadd_readvariableop_resource@
<encoder_conv1d_7_conv1d_expanddims_1_readvariableop_resource4
0encoder_conv1d_7_biasadd_readvariableop_resource2
.encoder_dense_6_matmul_readvariableop_resource3
/encoder_dense_6_biasadd_readvariableop_resource1
-encoder_z_mean_matmul_readvariableop_resource2
.encoder_z_mean_biasadd_readvariableop_resource4
0encoder_z_log_var_matmul_readvariableop_resource5
1encoder_z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2??*encoder/sampling_3/StatefulPartitionedCall?
&encoder/conv1d_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&encoder/conv1d_6/conv1d/ExpandDims/dim?
"encoder/conv1d_6/conv1d/ExpandDims
ExpandDimsinput_7/encoder/conv1d_6/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2$
"encoder/conv1d_6/conv1d/ExpandDims?
3encoder/conv1d_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<encoder_conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype025
3encoder/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp?
(encoder/conv1d_6/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(encoder/conv1d_6/conv1d/ExpandDims_1/dim?
$encoder/conv1d_6/conv1d/ExpandDims_1
ExpandDims;encoder/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp:value:01encoder/conv1d_6/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2&
$encoder/conv1d_6/conv1d/ExpandDims_1?
encoder/conv1d_6/conv1dConv2D+encoder/conv1d_6/conv1d/ExpandDims:output:0-encoder/conv1d_6/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????@*
paddingSAME*
strides
2
encoder/conv1d_6/conv1d?
encoder/conv1d_6/conv1d/SqueezeSqueeze encoder/conv1d_6/conv1d:output:0*
T0*,
_output_shapes
:??????????@*
squeeze_dims

?????????2!
encoder/conv1d_6/conv1d/Squeeze?
'encoder/conv1d_6/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv1d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'encoder/conv1d_6/BiasAdd/ReadVariableOp?
encoder/conv1d_6/BiasAddBiasAdd(encoder/conv1d_6/conv1d/Squeeze:output:0/encoder/conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2
encoder/conv1d_6/BiasAdd?
encoder/conv1d_6/ReluRelu!encoder/conv1d_6/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
encoder/conv1d_6/Relu?
&encoder/max_pooling1d_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&encoder/max_pooling1d_6/ExpandDims/dim?
"encoder/max_pooling1d_6/ExpandDims
ExpandDims#encoder/conv1d_6/Relu:activations:0/encoder/max_pooling1d_6/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????@2$
"encoder/max_pooling1d_6/ExpandDims?
encoder/max_pooling1d_6/MaxPoolMaxPool+encoder/max_pooling1d_6/ExpandDims:output:0*/
_output_shapes
:?????????C@*
ksize
*
paddingSAME*
strides
2!
encoder/max_pooling1d_6/MaxPool?
encoder/max_pooling1d_6/SqueezeSqueeze(encoder/max_pooling1d_6/MaxPool:output:0*
T0*+
_output_shapes
:?????????C@*
squeeze_dims
2!
encoder/max_pooling1d_6/Squeeze?
&encoder/conv1d_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&encoder/conv1d_7/conv1d/ExpandDims/dim?
"encoder/conv1d_7/conv1d/ExpandDims
ExpandDims(encoder/max_pooling1d_6/Squeeze:output:0/encoder/conv1d_7/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????C@2$
"encoder/conv1d_7/conv1d/ExpandDims?
3encoder/conv1d_7/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<encoder_conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype025
3encoder/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp?
(encoder/conv1d_7/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(encoder/conv1d_7/conv1d/ExpandDims_1/dim?
$encoder/conv1d_7/conv1d/ExpandDims_1
ExpandDims;encoder/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp:value:01encoder/conv1d_7/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2&
$encoder/conv1d_7/conv1d/ExpandDims_1?
encoder/conv1d_7/conv1dConv2D+encoder/conv1d_7/conv1d/ExpandDims:output:0-encoder/conv1d_7/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????C *
paddingSAME*
strides
2
encoder/conv1d_7/conv1d?
encoder/conv1d_7/conv1d/SqueezeSqueeze encoder/conv1d_7/conv1d:output:0*
T0*+
_output_shapes
:?????????C *
squeeze_dims

?????????2!
encoder/conv1d_7/conv1d/Squeeze?
'encoder/conv1d_7/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv1d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'encoder/conv1d_7/BiasAdd/ReadVariableOp?
encoder/conv1d_7/BiasAddBiasAdd(encoder/conv1d_7/conv1d/Squeeze:output:0/encoder/conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????C 2
encoder/conv1d_7/BiasAdd?
encoder/conv1d_7/ReluRelu!encoder/conv1d_7/BiasAdd:output:0*
T0*+
_output_shapes
:?????????C 2
encoder/conv1d_7/Relu?
&encoder/max_pooling1d_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&encoder/max_pooling1d_7/ExpandDims/dim?
"encoder/max_pooling1d_7/ExpandDims
ExpandDims#encoder/conv1d_7/Relu:activations:0/encoder/max_pooling1d_7/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????C 2$
"encoder/max_pooling1d_7/ExpandDims?
encoder/max_pooling1d_7/MaxPoolMaxPool+encoder/max_pooling1d_7/ExpandDims:output:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2!
encoder/max_pooling1d_7/MaxPool?
encoder/max_pooling1d_7/SqueezeSqueeze(encoder/max_pooling1d_7/MaxPool:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims
2!
encoder/max_pooling1d_7/Squeeze?
encoder/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
encoder/flatten_3/Const?
encoder/flatten_3/ReshapeReshape(encoder/max_pooling1d_7/Squeeze:output:0 encoder/flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????2
encoder/flatten_3/Reshape?
%encoder/dense_6/MatMul/ReadVariableOpReadVariableOp.encoder_dense_6_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02'
%encoder/dense_6/MatMul/ReadVariableOp?
encoder/dense_6/MatMulMatMul"encoder/flatten_3/Reshape:output:0-encoder/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
encoder/dense_6/MatMul?
&encoder/dense_6/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&encoder/dense_6/BiasAdd/ReadVariableOp?
encoder/dense_6/BiasAddBiasAdd encoder/dense_6/MatMul:product:0.encoder/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
encoder/dense_6/BiasAdd?
encoder/dense_6/ReluRelu encoder/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
encoder/dense_6/Relu?
$encoder/z_mean/MatMul/ReadVariableOpReadVariableOp-encoder_z_mean_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02&
$encoder/z_mean/MatMul/ReadVariableOp?
encoder/z_mean/MatMulMatMul"encoder/dense_6/Relu:activations:0,encoder/z_mean/MatMul/ReadVariableOp:value:0*
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
encoder/z_log_var/MatMulMatMul"encoder/dense_6/Relu:activations:0/encoder/z_log_var/MatMul/ReadVariableOp:value:0*
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
encoder/z_log_var/BiasAdd?
*encoder/sampling_3/StatefulPartitionedCallStatefulPartitionedCallencoder/z_mean/BiasAdd:output:0"encoder/z_log_var/BiasAdd:output:0*
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
GPU2*0J 8? *3
f.R,
*__inference_restored_function_body_36627712,
*encoder/sampling_3/StatefulPartitionedCall?
IdentityIdentity3encoder/sampling_3/StatefulPartitionedCall:output:0+^encoder/sampling_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity"encoder/z_log_var/BiasAdd:output:0+^encoder/sampling_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identityencoder/z_mean/BiasAdd:output:0+^encoder/sampling_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*S
_input_shapesB
@:??????????::::::::::2X
*encoder/sampling_3/StatefulPartitionedCall*encoder/sampling_3/StatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_7
?
?
E__inference_conv1d_6_layer_call_and_return_conditional_losses_5494441

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
?
?
E__inference_conv1d_7_layer_call_and_return_conditional_losses_5494474

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
?
h
L__inference_max_pooling1d_6_layer_call_and_return_conditional_losses_5494400

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
?
?
E__inference_conv1d_6_layer_call_and_return_conditional_losses_5494977

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
?
M
1__inference_max_pooling1d_7_layer_call_fn_5494421

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
L__inference_max_pooling1d_7_layer_call_and_return_conditional_losses_54944152
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
?
q
*__inference_restored_function_body_3662771

inputs
inputs_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*'
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
G__inference_sampling_3_layer_call_and_return_conditional_losses_36619572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_z_mean_layer_call_and_return_conditional_losses_5494542

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
?H
?
D__inference_encoder_layer_call_and_return_conditional_losses_5494903

inputs8
4conv1d_6_conv1d_expanddims_1_readvariableop_resource,
(conv1d_6_biasadd_readvariableop_resource8
4conv1d_7_conv1d_expanddims_1_readvariableop_resource,
(conv1d_7_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource)
%z_mean_matmul_readvariableop_resource*
&z_mean_biasadd_readvariableop_resource,
(z_log_var_matmul_readvariableop_resource-
)z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2??"sampling_3/StatefulPartitionedCall?
conv1d_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_6/conv1d/ExpandDims/dim?
conv1d_6/conv1d/ExpandDims
ExpandDimsinputs'conv1d_6/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d_6/conv1d/ExpandDims?
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02-
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_6/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_6/conv1d/ExpandDims_1/dim?
conv1d_6/conv1d/ExpandDims_1
ExpandDims3conv1d_6/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_6/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_6/conv1d/ExpandDims_1?
conv1d_6/conv1dConv2D#conv1d_6/conv1d/ExpandDims:output:0%conv1d_6/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????@*
paddingSAME*
strides
2
conv1d_6/conv1d?
conv1d_6/conv1d/SqueezeSqueezeconv1d_6/conv1d:output:0*
T0*,
_output_shapes
:??????????@*
squeeze_dims

?????????2
conv1d_6/conv1d/Squeeze?
conv1d_6/BiasAdd/ReadVariableOpReadVariableOp(conv1d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_6/BiasAdd/ReadVariableOp?
conv1d_6/BiasAddBiasAdd conv1d_6/conv1d/Squeeze:output:0'conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2
conv1d_6/BiasAddx
conv1d_6/ReluReluconv1d_6/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
conv1d_6/Relu?
max_pooling1d_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_6/ExpandDims/dim?
max_pooling1d_6/ExpandDims
ExpandDimsconv1d_6/Relu:activations:0'max_pooling1d_6/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????@2
max_pooling1d_6/ExpandDims?
max_pooling1d_6/MaxPoolMaxPool#max_pooling1d_6/ExpandDims:output:0*/
_output_shapes
:?????????C@*
ksize
*
paddingSAME*
strides
2
max_pooling1d_6/MaxPool?
max_pooling1d_6/SqueezeSqueeze max_pooling1d_6/MaxPool:output:0*
T0*+
_output_shapes
:?????????C@*
squeeze_dims
2
max_pooling1d_6/Squeeze?
conv1d_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_7/conv1d/ExpandDims/dim?
conv1d_7/conv1d/ExpandDims
ExpandDims max_pooling1d_6/Squeeze:output:0'conv1d_7/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????C@2
conv1d_7/conv1d/ExpandDims?
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02-
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_7/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_7/conv1d/ExpandDims_1/dim?
conv1d_7/conv1d/ExpandDims_1
ExpandDims3conv1d_7/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_7/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d_7/conv1d/ExpandDims_1?
conv1d_7/conv1dConv2D#conv1d_7/conv1d/ExpandDims:output:0%conv1d_7/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????C *
paddingSAME*
strides
2
conv1d_7/conv1d?
conv1d_7/conv1d/SqueezeSqueezeconv1d_7/conv1d:output:0*
T0*+
_output_shapes
:?????????C *
squeeze_dims

?????????2
conv1d_7/conv1d/Squeeze?
conv1d_7/BiasAdd/ReadVariableOpReadVariableOp(conv1d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_7/BiasAdd/ReadVariableOp?
conv1d_7/BiasAddBiasAdd conv1d_7/conv1d/Squeeze:output:0'conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????C 2
conv1d_7/BiasAddw
conv1d_7/ReluReluconv1d_7/BiasAdd:output:0*
T0*+
_output_shapes
:?????????C 2
conv1d_7/Relu?
max_pooling1d_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_7/ExpandDims/dim?
max_pooling1d_7/ExpandDims
ExpandDimsconv1d_7/Relu:activations:0'max_pooling1d_7/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????C 2
max_pooling1d_7/ExpandDims?
max_pooling1d_7/MaxPoolMaxPool#max_pooling1d_7/ExpandDims:output:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2
max_pooling1d_7/MaxPool?
max_pooling1d_7/SqueezeSqueeze max_pooling1d_7/MaxPool:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims
2
max_pooling1d_7/Squeezes
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_3/Const?
flatten_3/ReshapeReshape max_pooling1d_7/Squeeze:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_3/Reshape?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulflatten_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_6/BiasAddp
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_6/Relu?
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
z_mean/MatMul/ReadVariableOp?
z_mean/MatMulMatMuldense_6/Relu:activations:0$z_mean/MatMul/ReadVariableOp:value:0*
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
z_log_var/MatMulMatMuldense_6/Relu:activations:0'z_log_var/MatMul/ReadVariableOp:value:0*
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
z_log_var/BiasAdd?
"sampling_3/StatefulPartitionedCallStatefulPartitionedCallz_mean/BiasAdd:output:0z_log_var/BiasAdd:output:0*
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
GPU2*0J 8? *3
f.R,
*__inference_restored_function_body_36627712$
"sampling_3/StatefulPartitionedCall?
IdentityIdentityz_mean/BiasAdd:output:0#^sampling_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identityz_log_var/BiasAdd:output:0#^sampling_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity+sampling_3/StatefulPartitionedCall:output:0#^sampling_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*S
_input_shapesB
@:??????????::::::::::2H
"sampling_3/StatefulPartitionedCall"sampling_3/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_z_mean_layer_call_and_return_conditional_losses_5495052

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
?/
?
D__inference_encoder_layer_call_and_return_conditional_losses_5494661

inputs
conv1d_6_5494629
conv1d_6_5494631
conv1d_7_5494635
conv1d_7_5494637
dense_6_5494642
dense_6_5494644
z_mean_5494647
z_mean_5494649
z_log_var_5494652
z_log_var_5494654
identity

identity_1

identity_2?? conv1d_6/StatefulPartitionedCall? conv1d_7/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?"sampling_3/StatefulPartitionedCall?!z_log_var/StatefulPartitionedCall?z_mean/StatefulPartitionedCall?
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_6_5494629conv1d_6_5494631*
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
E__inference_conv1d_6_layer_call_and_return_conditional_losses_54944412"
 conv1d_6/StatefulPartitionedCall?
max_pooling1d_6/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling1d_6_layer_call_and_return_conditional_losses_54944002!
max_pooling1d_6/PartitionedCall?
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_6/PartitionedCall:output:0conv1d_7_5494635conv1d_7_5494637*
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
E__inference_conv1d_7_layer_call_and_return_conditional_losses_54944742"
 conv1d_7/StatefulPartitionedCall?
max_pooling1d_7/PartitionedCallPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0*
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
L__inference_max_pooling1d_7_layer_call_and_return_conditional_losses_54944152!
max_pooling1d_7/PartitionedCall?
flatten_3/PartitionedCallPartitionedCall(max_pooling1d_7/PartitionedCall:output:0*
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
F__inference_flatten_3_layer_call_and_return_conditional_losses_54944972
flatten_3/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_5494642dense_6_5494644*
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
D__inference_dense_6_layer_call_and_return_conditional_losses_54945162!
dense_6/StatefulPartitionedCall?
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0z_mean_5494647z_mean_5494649*
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
C__inference_z_mean_layer_call_and_return_conditional_losses_54945422 
z_mean/StatefulPartitionedCall?
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0z_log_var_5494652z_log_var_5494654*
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
F__inference_z_log_var_layer_call_and_return_conditional_losses_54945682#
!z_log_var/StatefulPartitionedCall?
"sampling_3/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *3
f.R,
*__inference_restored_function_body_36627712$
"sampling_3/StatefulPartitionedCall?
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall#^sampling_3/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall#^sampling_3/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity+sampling_3/StatefulPartitionedCall:output:0!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall#^sampling_3/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*S
_input_shapesB
@:??????????::::::::::2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2H
"sampling_3/StatefulPartitionedCall"sampling_3/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?-
?
#__inference__traced_restore_5495175
file_prefix$
 assignvariableop_conv1d_6_kernel$
 assignvariableop_1_conv1d_6_bias&
"assignvariableop_2_conv1d_7_kernel$
 assignvariableop_3_conv1d_7_bias%
!assignvariableop_4_dense_6_kernel#
assignvariableop_5_dense_6_bias$
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
AssignVariableOpAssignVariableOp assignvariableop_conv1d_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_6_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_7_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_7_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_6_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_6_biasIdentity_5:output:0"/device:CPU:0*
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
?
u
,__inference_sampling_3_layer_call_fn_3662194
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
G__inference_sampling_3_layer_call_and_return_conditional_losses_36621882
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
?
?
F__inference_z_log_var_layer_call_and_return_conditional_losses_5495071

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
?
?
+__inference_z_log_var_layer_call_fn_5495080

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
F__inference_z_log_var_layer_call_and_return_conditional_losses_54945682
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
F__inference_z_log_var_layer_call_and_return_conditional_losses_5494568

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
?
t
G__inference_sampling_3_layer_call_and_return_conditional_losses_3662188

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
seed2???2$
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
?
M
1__inference_max_pooling1d_6_layer_call_fn_5494406

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
L__inference_max_pooling1d_6_layer_call_and_return_conditional_losses_54944002
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
?
?
D__inference_dense_6_layer_call_and_return_conditional_losses_5494516

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
?
?
E__inference_conv1d_7_layer_call_and_return_conditional_losses_5495002

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
?H
?
D__inference_encoder_layer_call_and_return_conditional_losses_5494843

inputs8
4conv1d_6_conv1d_expanddims_1_readvariableop_resource,
(conv1d_6_biasadd_readvariableop_resource8
4conv1d_7_conv1d_expanddims_1_readvariableop_resource,
(conv1d_7_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource)
%z_mean_matmul_readvariableop_resource*
&z_mean_biasadd_readvariableop_resource,
(z_log_var_matmul_readvariableop_resource-
)z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2??"sampling_3/StatefulPartitionedCall?
conv1d_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_6/conv1d/ExpandDims/dim?
conv1d_6/conv1d/ExpandDims
ExpandDimsinputs'conv1d_6/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d_6/conv1d/ExpandDims?
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02-
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_6/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_6/conv1d/ExpandDims_1/dim?
conv1d_6/conv1d/ExpandDims_1
ExpandDims3conv1d_6/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_6/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_6/conv1d/ExpandDims_1?
conv1d_6/conv1dConv2D#conv1d_6/conv1d/ExpandDims:output:0%conv1d_6/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????@*
paddingSAME*
strides
2
conv1d_6/conv1d?
conv1d_6/conv1d/SqueezeSqueezeconv1d_6/conv1d:output:0*
T0*,
_output_shapes
:??????????@*
squeeze_dims

?????????2
conv1d_6/conv1d/Squeeze?
conv1d_6/BiasAdd/ReadVariableOpReadVariableOp(conv1d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_6/BiasAdd/ReadVariableOp?
conv1d_6/BiasAddBiasAdd conv1d_6/conv1d/Squeeze:output:0'conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2
conv1d_6/BiasAddx
conv1d_6/ReluReluconv1d_6/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
conv1d_6/Relu?
max_pooling1d_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_6/ExpandDims/dim?
max_pooling1d_6/ExpandDims
ExpandDimsconv1d_6/Relu:activations:0'max_pooling1d_6/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????@2
max_pooling1d_6/ExpandDims?
max_pooling1d_6/MaxPoolMaxPool#max_pooling1d_6/ExpandDims:output:0*/
_output_shapes
:?????????C@*
ksize
*
paddingSAME*
strides
2
max_pooling1d_6/MaxPool?
max_pooling1d_6/SqueezeSqueeze max_pooling1d_6/MaxPool:output:0*
T0*+
_output_shapes
:?????????C@*
squeeze_dims
2
max_pooling1d_6/Squeeze?
conv1d_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_7/conv1d/ExpandDims/dim?
conv1d_7/conv1d/ExpandDims
ExpandDims max_pooling1d_6/Squeeze:output:0'conv1d_7/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????C@2
conv1d_7/conv1d/ExpandDims?
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02-
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_7/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_7/conv1d/ExpandDims_1/dim?
conv1d_7/conv1d/ExpandDims_1
ExpandDims3conv1d_7/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_7/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d_7/conv1d/ExpandDims_1?
conv1d_7/conv1dConv2D#conv1d_7/conv1d/ExpandDims:output:0%conv1d_7/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????C *
paddingSAME*
strides
2
conv1d_7/conv1d?
conv1d_7/conv1d/SqueezeSqueezeconv1d_7/conv1d:output:0*
T0*+
_output_shapes
:?????????C *
squeeze_dims

?????????2
conv1d_7/conv1d/Squeeze?
conv1d_7/BiasAdd/ReadVariableOpReadVariableOp(conv1d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_7/BiasAdd/ReadVariableOp?
conv1d_7/BiasAddBiasAdd conv1d_7/conv1d/Squeeze:output:0'conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????C 2
conv1d_7/BiasAddw
conv1d_7/ReluReluconv1d_7/BiasAdd:output:0*
T0*+
_output_shapes
:?????????C 2
conv1d_7/Relu?
max_pooling1d_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_7/ExpandDims/dim?
max_pooling1d_7/ExpandDims
ExpandDimsconv1d_7/Relu:activations:0'max_pooling1d_7/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????C 2
max_pooling1d_7/ExpandDims?
max_pooling1d_7/MaxPoolMaxPool#max_pooling1d_7/ExpandDims:output:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2
max_pooling1d_7/MaxPool?
max_pooling1d_7/SqueezeSqueeze max_pooling1d_7/MaxPool:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims
2
max_pooling1d_7/Squeezes
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_3/Const?
flatten_3/ReshapeReshape max_pooling1d_7/Squeeze:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_3/Reshape?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulflatten_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_6/BiasAddp
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_6/Relu?
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
z_mean/MatMul/ReadVariableOp?
z_mean/MatMulMatMuldense_6/Relu:activations:0$z_mean/MatMul/ReadVariableOp:value:0*
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
z_log_var/MatMulMatMuldense_6/Relu:activations:0'z_log_var/MatMul/ReadVariableOp:value:0*
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
z_log_var/BiasAdd?
"sampling_3/StatefulPartitionedCallStatefulPartitionedCallz_mean/BiasAdd:output:0z_log_var/BiasAdd:output:0*
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
GPU2*0J 8? *3
f.R,
*__inference_restored_function_body_36627712$
"sampling_3/StatefulPartitionedCall?
IdentityIdentityz_mean/BiasAdd:output:0#^sampling_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identityz_log_var/BiasAdd:output:0#^sampling_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity+sampling_3/StatefulPartitionedCall:output:0#^sampling_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*S
_input_shapesB
@:??????????::::::::::2H
"sampling_3/StatefulPartitionedCall"sampling_3/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

*__inference_conv1d_7_layer_call_fn_5495011

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
E__inference_conv1d_7_layer_call_and_return_conditional_losses_54944742
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
?
?
)__inference_encoder_layer_call_fn_5494752
input_7
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
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
D__inference_encoder_layer_call_and_return_conditional_losses_54947252
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
_user_specified_name	input_7
?
?
D__inference_dense_6_layer_call_and_return_conditional_losses_5495033

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
?
?
)__inference_encoder_layer_call_fn_5494961

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
D__inference_encoder_layer_call_and_return_conditional_losses_54947252
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
?
b
F__inference_flatten_3_layer_call_and_return_conditional_losses_5494497

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
input_75
serving_default_input_7:0??????????>

sampling_30
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
#_self_saveable_object_factories

signatures
regularization_losses
trainable_variables
	variables
	keras_api
*{&call_and_return_all_conditional_losses
|__call__
}_default_save_signature"?G
_tf_keras_network?G{"class_name": "Functional", "name": "encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 201, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_6", "inbound_nodes": [[["input_7", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_6", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "same", "data_format": "channels_last"}, "name": "max_pooling1d_6", "inbound_nodes": [[["conv1d_6", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_7", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_7", "inbound_nodes": [[["max_pooling1d_6", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_7", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "same", "data_format": "channels_last"}, "name": "max_pooling1d_7", "inbound_nodes": [[["conv1d_7", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["max_pooling1d_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["flatten_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_var", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "sampling_3", "trainable": true, "dtype": "float32"}, "name": "sampling_3", "inbound_nodes": [[["z_mean", 0, 0, {}], ["z_log_var", 0, 0, {}]]]}], "input_layers": [["input_7", 0, 0]], "output_layers": [["z_mean", 0, 0], ["z_log_var", 0, 0], ["sampling_3", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 201, 6]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 201, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv1d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_6", "inbound_nodes": [[["input_7", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_6", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "same", "data_format": "channels_last"}, "name": "max_pooling1d_6", "inbound_nodes": [[["conv1d_6", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_7", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_7", "inbound_nodes": [[["max_pooling1d_6", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "max_pooling1d_7", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "same", "data_format": "channels_last"}, "name": "max_pooling1d_7", "inbound_nodes": [[["conv1d_7", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["max_pooling1d_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["flatten_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_var", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "sampling_3", "trainable": true, "dtype": "float32"}, "name": "sampling_3", "inbound_nodes": [[["z_mean", 0, 0, {}], ["z_log_var", 0, 0, {}]]]}], "input_layers": [["input_7", 0, 0]], "output_layers": [["z_mean", 0, 0], ["z_log_var", 0, 0], ["sampling_3", 0, 0]]}}}
?
#_self_saveable_object_factories"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_7", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 201, 6]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 201, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}}
?


kernel
bias
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
*~&call_and_return_all_conditional_losses
__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 201, 6]}}
?
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling1D", "name": "max_pooling1d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d_6", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "same", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


kernel
bias
# _self_saveable_object_factories
!regularization_losses
"trainable_variables
#	variables
$	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_7", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 67, 64]}}
?
#%_self_saveable_object_factories
&regularization_losses
'trainable_variables
(	variables
)	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling1D", "name": "max_pooling1d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling1d_7", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [3]}, "pool_size": {"class_name": "__tuple__", "items": [3]}, "padding": "same", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
#*_self_saveable_object_factories
+regularization_losses
,trainable_variables
-	variables
.	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

/kernel
0bias
#1_self_saveable_object_factories
2regularization_losses
3trainable_variables
4	variables
5	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 736}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 736]}}
?

6kernel
7bias
#8_self_saveable_object_factories
9regularization_losses
:trainable_variables
;	variables
<	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "z_mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?

=kernel
>bias
#?_self_saveable_object_factories
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "z_log_var", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
#D_self_saveable_object_factories
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Sampling", "name": "sampling_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "sampling_3", "trainable": true, "dtype": "float32"}}
 "
trackable_dict_wrapper
-
?serving_default"
signature_map
 "
trackable_list_wrapper
f
0
1
2
3
/4
05
66
77
=8
>9"
trackable_list_wrapper
f
0
1
2
3
/4
05
66
77
=8
>9"
trackable_list_wrapper
?
Imetrics

Jlayers
regularization_losses
trainable_variables
	variables
Klayer_regularization_losses
Lnon_trainable_variables
Mlayer_metrics
|__call__
}_default_save_signature
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
%:#@2conv1d_6/kernel
:@2conv1d_6/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Nmetrics

Olayers
regularization_losses
trainable_variables
	variables
Player_regularization_losses
Qnon_trainable_variables
Rlayer_metrics
__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Smetrics

Tlayers
regularization_losses
trainable_variables
	variables
Ulayer_regularization_losses
Vnon_trainable_variables
Wlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#@ 2conv1d_7/kernel
: 2conv1d_7/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Xmetrics

Ylayers
!regularization_losses
"trainable_variables
#	variables
Zlayer_regularization_losses
[non_trainable_variables
\layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
]metrics

^layers
&regularization_losses
'trainable_variables
(	variables
_layer_regularization_losses
`non_trainable_variables
alayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
bmetrics

clayers
+regularization_losses
,trainable_variables
-	variables
dlayer_regularization_losses
enon_trainable_variables
flayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?@2dense_6/kernel
:@2dense_6/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
?
gmetrics

hlayers
2regularization_losses
3trainable_variables
4	variables
ilayer_regularization_losses
jnon_trainable_variables
klayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:@2z_mean/kernel
:2z_mean/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
?
lmetrics

mlayers
9regularization_losses
:trainable_variables
;	variables
nlayer_regularization_losses
onon_trainable_variables
player_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": @2z_log_var/kernel
:2z_log_var/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
?
qmetrics

rlayers
@regularization_losses
Atrainable_variables
B	variables
slayer_regularization_losses
tnon_trainable_variables
ulayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
vmetrics

wlayers
Eregularization_losses
Ftrainable_variables
G	variables
xlayer_regularization_losses
ynon_trainable_variables
zlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
D__inference_encoder_layer_call_and_return_conditional_losses_5494903
D__inference_encoder_layer_call_and_return_conditional_losses_5494588
D__inference_encoder_layer_call_and_return_conditional_losses_5494623
D__inference_encoder_layer_call_and_return_conditional_losses_5494843?
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
)__inference_encoder_layer_call_fn_5494961
)__inference_encoder_layer_call_fn_5494688
)__inference_encoder_layer_call_fn_5494752
)__inference_encoder_layer_call_fn_5494932?
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
"__inference__wrapped_model_5494391?
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
input_7??????????
?2?
E__inference_conv1d_6_layer_call_and_return_conditional_losses_5494977?
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
*__inference_conv1d_6_layer_call_fn_5494986?
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
L__inference_max_pooling1d_6_layer_call_and_return_conditional_losses_5494400?
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
1__inference_max_pooling1d_6_layer_call_fn_5494406?
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
E__inference_conv1d_7_layer_call_and_return_conditional_losses_5495002?
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
*__inference_conv1d_7_layer_call_fn_5495011?
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
L__inference_max_pooling1d_7_layer_call_and_return_conditional_losses_5494415?
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
1__inference_max_pooling1d_7_layer_call_fn_5494421?
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
F__inference_flatten_3_layer_call_and_return_conditional_losses_5495017?
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
+__inference_flatten_3_layer_call_fn_5495022?
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
D__inference_dense_6_layer_call_and_return_conditional_losses_5495033?
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
)__inference_dense_6_layer_call_fn_5495042?
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
C__inference_z_mean_layer_call_and_return_conditional_losses_5495052?
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
(__inference_z_mean_layer_call_fn_5495061?
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
F__inference_z_log_var_layer_call_and_return_conditional_losses_5495071?
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
+__inference_z_log_var_layer_call_fn_5495080?
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
G__inference_sampling_3_layer_call_and_return_conditional_losses_3661957?
???
FullArgSpec
args?

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
annotations? *
 
?2?
,__inference_sampling_3_layer_call_fn_3662194?
???
FullArgSpec
args?

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
annotations? *
 
4B2
%__inference_signature_wrapper_5494783input_7?
"__inference__wrapped_model_5494391?
/067=>5?2
+?(
&?#
input_7??????????
? "???
2

sampling_3$?!

sampling_3?????????
0
	z_log_var#? 
	z_log_var?????????
*
z_mean ?
z_mean??????????
E__inference_conv1d_6_layer_call_and_return_conditional_losses_5494977f4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????@
? ?
*__inference_conv1d_6_layer_call_fn_5494986Y4?1
*?'
%?"
inputs??????????
? "???????????@?
E__inference_conv1d_7_layer_call_and_return_conditional_losses_5495002d3?0
)?&
$?!
inputs?????????C@
? ")?&
?
0?????????C 
? ?
*__inference_conv1d_7_layer_call_fn_5495011W3?0
)?&
$?!
inputs?????????C@
? "??????????C ?
D__inference_dense_6_layer_call_and_return_conditional_losses_5495033]/00?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? }
)__inference_dense_6_layer_call_fn_5495042P/00?-
&?#
!?
inputs??????????
? "??????????@?
D__inference_encoder_layer_call_and_return_conditional_losses_5494588?
/067=>=?:
3?0
&?#
input_7??????????
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
D__inference_encoder_layer_call_and_return_conditional_losses_5494623?
/067=>=?:
3?0
&?#
input_7??????????
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
D__inference_encoder_layer_call_and_return_conditional_losses_5494843?
/067=><?9
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
D__inference_encoder_layer_call_and_return_conditional_losses_5494903?
/067=><?9
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
)__inference_encoder_layer_call_fn_5494688?
/067=>=?:
3?0
&?#
input_7??????????
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
)__inference_encoder_layer_call_fn_5494752?
/067=>=?:
3?0
&?#
input_7??????????
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
)__inference_encoder_layer_call_fn_5494932?
/067=><?9
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
)__inference_encoder_layer_call_fn_5494961?
/067=><?9
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
F__inference_flatten_3_layer_call_and_return_conditional_losses_5495017]3?0
)?&
$?!
inputs????????? 
? "&?#
?
0??????????
? 
+__inference_flatten_3_layer_call_fn_5495022P3?0
)?&
$?!
inputs????????? 
? "????????????
L__inference_max_pooling1d_6_layer_call_and_return_conditional_losses_5494400?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
1__inference_max_pooling1d_6_layer_call_fn_5494406wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
L__inference_max_pooling1d_7_layer_call_and_return_conditional_losses_5494415?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
1__inference_max_pooling1d_7_layer_call_fn_5494421wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
G__inference_sampling_3_layer_call_and_return_conditional_losses_3661957?Z?W
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
,__inference_sampling_3_layer_call_fn_3662194vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
%__inference_signature_wrapper_5494783?
/067=>@?=
? 
6?3
1
input_7&?#
input_7??????????"???
2

sampling_3$?!

sampling_3?????????
0
	z_log_var#? 
	z_log_var?????????
*
z_mean ?
z_mean??????????
F__inference_z_log_var_layer_call_and_return_conditional_losses_5495071\=>/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? ~
+__inference_z_log_var_layer_call_fn_5495080O=>/?,
%?"
 ?
inputs?????????@
? "???????????
C__inference_z_mean_layer_call_and_return_conditional_losses_5495052\67/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? {
(__inference_z_mean_layer_call_fn_5495061O67/?,
%?"
 ?
inputs?????????@
? "??????????