OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
creg c[24];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
h q[16];
h q[17];
h q[18];
h q[19];
h q[20];
h q[21];
h q[22];
h q[23];
cx q[0],q[13];
u1(0.3335479283054058) q[13];
cx q[0],q[13];
cx q[0],q[14];
u1(0.3335479283054058) q[14];
cx q[0],q[14];
cx q[0],q[15];
u1(0.3335479283054058) q[15];
cx q[0],q[15];
cx q[0],q[17];
u1(0.3335479283054058) q[17];
cx q[0],q[17];
cx q[0],q[18];
u1(0.3335479283054058) q[18];
cx q[0],q[18];
cx q[0],q[19];
u1(0.3335479283054058) q[19];
cx q[0],q[19];
cx q[0],q[21];
u1(0.3335479283054058) q[21];
cx q[0],q[21];
cx q[0],q[22];
u1(0.3335479283054058) q[22];
cx q[0],q[22];
cx q[1],q[4];
u1(0.3335479283054058) q[4];
cx q[1],q[4];
cx q[1],q[6];
u1(0.3335479283054058) q[6];
cx q[1],q[6];
cx q[1],q[7];
u1(0.3335479283054058) q[7];
cx q[1],q[7];
cx q[1],q[13];
u1(0.3335479283054058) q[13];
cx q[1],q[13];
cx q[1],q[14];
u1(0.3335479283054058) q[14];
cx q[1],q[14];
cx q[1],q[19];
u1(0.3335479283054058) q[19];
cx q[1],q[19];
cx q[2],q[9];
u1(0.3335479283054058) q[9];
cx q[2],q[9];
cx q[2],q[16];
u1(0.3335479283054058) q[16];
cx q[2],q[16];
cx q[2],q[21];
u1(0.3335479283054058) q[21];
cx q[2],q[21];
cx q[3],q[6];
u1(0.3335479283054058) q[6];
cx q[3],q[6];
cx q[3],q[9];
u1(0.3335479283054058) q[9];
cx q[3],q[9];
cx q[3],q[10];
u1(0.3335479283054058) q[10];
cx q[3],q[10];
cx q[3],q[17];
u1(0.3335479283054058) q[17];
cx q[3],q[17];
cx q[3],q[23];
u1(0.3335479283054058) q[23];
cx q[3],q[23];
cx q[4],q[5];
u1(0.3335479283054058) q[5];
cx q[4],q[5];
cx q[4],q[6];
u1(0.3335479283054058) q[6];
cx q[4],q[6];
cx q[4],q[11];
u1(0.3335479283054058) q[11];
cx q[4],q[11];
cx q[4],q[12];
u1(0.3335479283054058) q[12];
cx q[4],q[12];
cx q[4],q[13];
u1(0.3335479283054058) q[13];
cx q[4],q[13];
cx q[4],q[16];
u1(0.3335479283054058) q[16];
cx q[4],q[16];
cx q[4],q[19];
u1(0.3335479283054058) q[19];
cx q[4],q[19];
cx q[5],q[6];
u1(0.3335479283054058) q[6];
cx q[5],q[6];
cx q[5],q[9];
u1(0.3335479283054058) q[9];
cx q[5],q[9];
cx q[5],q[10];
u1(0.3335479283054058) q[10];
cx q[5],q[10];
cx q[5],q[12];
u1(0.3335479283054058) q[12];
cx q[5],q[12];
cx q[5],q[14];
u1(0.3335479283054058) q[14];
cx q[5],q[14];
cx q[5],q[15];
u1(0.3335479283054058) q[15];
cx q[5],q[15];
cx q[5],q[17];
u1(0.3335479283054058) q[17];
cx q[5],q[17];
cx q[5],q[18];
u1(0.3335479283054058) q[18];
cx q[5],q[18];
cx q[5],q[19];
u1(0.3335479283054058) q[19];
cx q[5],q[19];
cx q[5],q[20];
u1(0.3335479283054058) q[20];
cx q[5],q[20];
cx q[6],q[9];
u1(0.3335479283054058) q[9];
cx q[6],q[9];
cx q[6],q[15];
u1(0.3335479283054058) q[15];
cx q[6],q[15];
cx q[6],q[23];
u1(0.3335479283054058) q[23];
cx q[6],q[23];
cx q[7],q[13];
u1(0.3335479283054058) q[13];
cx q[7],q[13];
cx q[7],q[15];
u1(0.3335479283054058) q[15];
cx q[7],q[15];
cx q[7],q[16];
u1(0.3335479283054058) q[16];
cx q[7],q[16];
cx q[7],q[17];
u1(0.3335479283054058) q[17];
cx q[7],q[17];
cx q[8],q[13];
u1(0.3335479283054058) q[13];
cx q[8],q[13];
cx q[8],q[20];
u1(0.3335479283054058) q[20];
cx q[8],q[20];
cx q[9],q[10];
u1(0.3335479283054058) q[10];
cx q[9],q[10];
cx q[10],q[13];
u1(0.3335479283054058) q[13];
cx q[10],q[13];
cx q[10],q[17];
u1(0.3335479283054058) q[17];
cx q[10],q[17];
cx q[10],q[18];
u1(0.3335479283054058) q[18];
cx q[10],q[18];
cx q[10],q[20];
u1(0.3335479283054058) q[20];
cx q[10],q[20];
cx q[11],q[14];
u1(0.3335479283054058) q[14];
cx q[11],q[14];
cx q[11],q[15];
u1(0.3335479283054058) q[15];
cx q[11],q[15];
cx q[11],q[16];
u1(0.3335479283054058) q[16];
cx q[11],q[16];
cx q[11],q[18];
u1(0.3335479283054058) q[18];
cx q[11],q[18];
cx q[11],q[22];
u1(0.3335479283054058) q[22];
cx q[11],q[22];
cx q[12],q[18];
u1(0.3335479283054058) q[18];
cx q[12],q[18];
cx q[12],q[19];
u1(0.3335479283054058) q[19];
cx q[12],q[19];
cx q[13],q[14];
u1(0.3335479283054058) q[14];
cx q[13],q[14];
cx q[13],q[15];
u1(0.3335479283054058) q[15];
cx q[13],q[15];
cx q[13],q[16];
u1(0.3335479283054058) q[16];
cx q[13],q[16];
cx q[13],q[21];
u1(0.3335479283054058) q[21];
cx q[13],q[21];
cx q[13],q[23];
u1(0.3335479283054058) q[23];
cx q[13],q[23];
cx q[14],q[17];
u1(0.3335479283054058) q[17];
cx q[14],q[17];
cx q[14],q[23];
u1(0.3335479283054058) q[23];
cx q[14],q[23];
cx q[15],q[18];
u1(0.3335479283054058) q[18];
cx q[15],q[18];
cx q[15],q[22];
u1(0.3335479283054058) q[22];
cx q[15],q[22];
cx q[18],q[20];
u1(0.3335479283054058) q[20];
cx q[18],q[20];
cx q[18],q[21];
u1(0.3335479283054058) q[21];
cx q[18],q[21];
cx q[19],q[20];
u1(0.3335479283054058) q[20];
cx q[19],q[20];
cx q[19],q[21];
u1(0.3335479283054058) q[21];
cx q[19],q[21];
cx q[19],q[22];
u1(0.3335479283054058) q[22];
cx q[19],q[22];
cx q[20],q[22];
u1(0.3335479283054058) q[22];
cx q[20],q[22];
cx q[20],q[23];
u1(0.3335479283054058) q[23];
cx q[20],q[23];
cx q[21],q[22];
u1(0.3335479283054058) q[22];
cx q[21],q[22];
cx q[22],q[23];
u1(0.3335479283054058) q[23];
cx q[22],q[23];
h q[0];
rz(0.8189527629853679) q[0];
h q[0];
h q[1];
rz(0.8189527629853679) q[1];
h q[1];
h q[2];
rz(0.8189527629853679) q[2];
h q[2];
h q[3];
rz(0.8189527629853679) q[3];
h q[3];
h q[4];
rz(0.8189527629853679) q[4];
h q[4];
h q[5];
rz(0.8189527629853679) q[5];
h q[5];
h q[6];
rz(0.8189527629853679) q[6];
h q[6];
h q[7];
rz(0.8189527629853679) q[7];
h q[7];
h q[8];
rz(0.8189527629853679) q[8];
h q[8];
h q[9];
rz(0.8189527629853679) q[9];
h q[9];
h q[10];
rz(0.8189527629853679) q[10];
h q[10];
h q[11];
rz(0.8189527629853679) q[11];
h q[11];
h q[12];
rz(0.8189527629853679) q[12];
h q[12];
h q[13];
rz(0.8189527629853679) q[13];
h q[13];
h q[14];
rz(0.8189527629853679) q[14];
h q[14];
h q[15];
rz(0.8189527629853679) q[15];
h q[15];
h q[16];
rz(0.8189527629853679) q[16];
h q[16];
h q[17];
rz(0.8189527629853679) q[17];
h q[17];
h q[18];
rz(0.8189527629853679) q[18];
h q[18];
h q[19];
rz(0.8189527629853679) q[19];
h q[19];
h q[20];
rz(0.8189527629853679) q[20];
h q[20];
h q[21];
rz(0.8189527629853679) q[21];
h q[21];
h q[22];
rz(0.8189527629853679) q[22];
h q[22];
h q[23];
rz(0.8189527629853679) q[23];
h q[23];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];
measure q[10] -> c[10];
measure q[11] -> c[11];
measure q[12] -> c[12];
measure q[13] -> c[13];
measure q[14] -> c[14];
measure q[15] -> c[15];
measure q[16] -> c[16];
measure q[17] -> c[17];
measure q[18] -> c[18];
measure q[19] -> c[19];
measure q[20] -> c[20];
measure q[21] -> c[21];
measure q[22] -> c[22];
measure q[23] -> c[23];
