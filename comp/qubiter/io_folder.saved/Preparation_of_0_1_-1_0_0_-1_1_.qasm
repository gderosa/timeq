// Name of Experiment: Preparation of {0,1,-1,0,0,-1,1} v25
// Description: An eigenstate of J

OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

x q[0];
h q[1];
x q[2];
h q[0];
cx q[1],q[2];
z q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
