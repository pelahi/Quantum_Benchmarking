OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
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
cx q[0],q[1];
u1(0.5017119571514081) q[1];
cx q[0],q[1];
cx q[0],q[2];
u1(0.5017119571514081) q[2];
cx q[0],q[2];
cx q[0],q[3];
u1(0.5017119571514081) q[3];
cx q[0],q[3];
cx q[0],q[5];
u1(0.5017119571514081) q[5];
cx q[0],q[5];
cx q[0],q[7];
u1(0.5017119571514081) q[7];
cx q[0],q[7];
cx q[0],q[8];
u1(0.5017119571514081) q[8];
cx q[0],q[8];
cx q[0],q[9];
u1(0.5017119571514081) q[9];
cx q[0],q[9];
cx q[0],q[10];
u1(0.5017119571514081) q[10];
cx q[0],q[10];
cx q[0],q[12];
u1(0.5017119571514081) q[12];
cx q[0],q[12];
cx q[0],q[14];
u1(0.5017119571514081) q[14];
cx q[0],q[14];
cx q[0],q[16];
u1(0.5017119571514081) q[16];
cx q[0],q[16];
cx q[0],q[17];
u1(0.5017119571514081) q[17];
cx q[0],q[17];
cx q[0],q[19];
u1(0.5017119571514081) q[19];
cx q[0],q[19];
cx q[1],q[2];
u1(0.5017119571514081) q[2];
cx q[1],q[2];
cx q[1],q[4];
u1(0.5017119571514081) q[4];
cx q[1],q[4];
cx q[1],q[7];
u1(0.5017119571514081) q[7];
cx q[1],q[7];
cx q[1],q[9];
u1(0.5017119571514081) q[9];
cx q[1],q[9];
cx q[1],q[12];
u1(0.5017119571514081) q[12];
cx q[1],q[12];
cx q[1],q[13];
u1(0.5017119571514081) q[13];
cx q[1],q[13];
cx q[1],q[14];
u1(0.5017119571514081) q[14];
cx q[1],q[14];
cx q[1],q[18];
u1(0.5017119571514081) q[18];
cx q[1],q[18];
cx q[2],q[3];
u1(0.5017119571514081) q[3];
cx q[2],q[3];
cx q[2],q[4];
u1(0.5017119571514081) q[4];
cx q[2],q[4];
cx q[2],q[8];
u1(0.5017119571514081) q[8];
cx q[2],q[8];
cx q[2],q[10];
u1(0.5017119571514081) q[10];
cx q[2],q[10];
cx q[2],q[11];
u1(0.5017119571514081) q[11];
cx q[2],q[11];
cx q[2],q[13];
u1(0.5017119571514081) q[13];
cx q[2],q[13];
cx q[2],q[14];
u1(0.5017119571514081) q[14];
cx q[2],q[14];
cx q[2],q[15];
u1(0.5017119571514081) q[15];
cx q[2],q[15];
cx q[2],q[18];
u1(0.5017119571514081) q[18];
cx q[2],q[18];
cx q[2],q[19];
u1(0.5017119571514081) q[19];
cx q[2],q[19];
cx q[3],q[4];
u1(0.5017119571514081) q[4];
cx q[3],q[4];
cx q[3],q[5];
u1(0.5017119571514081) q[5];
cx q[3],q[5];
cx q[3],q[6];
u1(0.5017119571514081) q[6];
cx q[3],q[6];
cx q[3],q[8];
u1(0.5017119571514081) q[8];
cx q[3],q[8];
cx q[3],q[9];
u1(0.5017119571514081) q[9];
cx q[3],q[9];
cx q[3],q[16];
u1(0.5017119571514081) q[16];
cx q[3],q[16];
cx q[3],q[19];
u1(0.5017119571514081) q[19];
cx q[3],q[19];
cx q[4],q[5];
u1(0.5017119571514081) q[5];
cx q[4],q[5];
cx q[4],q[6];
u1(0.5017119571514081) q[6];
cx q[4],q[6];
cx q[4],q[7];
u1(0.5017119571514081) q[7];
cx q[4],q[7];
cx q[4],q[9];
u1(0.5017119571514081) q[9];
cx q[4],q[9];
cx q[4],q[10];
u1(0.5017119571514081) q[10];
cx q[4],q[10];
cx q[4],q[14];
u1(0.5017119571514081) q[14];
cx q[4],q[14];
cx q[4],q[16];
u1(0.5017119571514081) q[16];
cx q[4],q[16];
cx q[4],q[17];
u1(0.5017119571514081) q[17];
cx q[4],q[17];
cx q[4],q[19];
u1(0.5017119571514081) q[19];
cx q[4],q[19];
cx q[5],q[6];
u1(0.5017119571514081) q[6];
cx q[5],q[6];
cx q[5],q[8];
u1(0.5017119571514081) q[8];
cx q[5],q[8];
cx q[5],q[9];
u1(0.5017119571514081) q[9];
cx q[5],q[9];
cx q[5],q[12];
u1(0.5017119571514081) q[12];
cx q[5],q[12];
cx q[5],q[13];
u1(0.5017119571514081) q[13];
cx q[5],q[13];
cx q[5],q[19];
u1(0.5017119571514081) q[19];
cx q[5],q[19];
cx q[6],q[8];
u1(0.5017119571514081) q[8];
cx q[6],q[8];
cx q[6],q[14];
u1(0.5017119571514081) q[14];
cx q[6],q[14];
cx q[6],q[15];
u1(0.5017119571514081) q[15];
cx q[6],q[15];
cx q[6],q[18];
u1(0.5017119571514081) q[18];
cx q[6],q[18];
cx q[7],q[8];
u1(0.5017119571514081) q[8];
cx q[7],q[8];
cx q[7],q[9];
u1(0.5017119571514081) q[9];
cx q[7],q[9];
cx q[7],q[10];
u1(0.5017119571514081) q[10];
cx q[7],q[10];
cx q[7],q[11];
u1(0.5017119571514081) q[11];
cx q[7],q[11];
cx q[7],q[13];
u1(0.5017119571514081) q[13];
cx q[7],q[13];
cx q[7],q[14];
u1(0.5017119571514081) q[14];
cx q[7],q[14];
cx q[7],q[18];
u1(0.5017119571514081) q[18];
cx q[7],q[18];
cx q[8],q[11];
u1(0.5017119571514081) q[11];
cx q[8],q[11];
cx q[8],q[12];
u1(0.5017119571514081) q[12];
cx q[8],q[12];
cx q[8],q[13];
u1(0.5017119571514081) q[13];
cx q[8],q[13];
cx q[8],q[15];
u1(0.5017119571514081) q[15];
cx q[8],q[15];
cx q[8],q[16];
u1(0.5017119571514081) q[16];
cx q[8],q[16];
cx q[8],q[17];
u1(0.5017119571514081) q[17];
cx q[8],q[17];
cx q[8],q[18];
u1(0.5017119571514081) q[18];
cx q[8],q[18];
cx q[9],q[11];
u1(0.5017119571514081) q[11];
cx q[9],q[11];
cx q[9],q[12];
u1(0.5017119571514081) q[12];
cx q[9],q[12];
cx q[9],q[14];
u1(0.5017119571514081) q[14];
cx q[9],q[14];
cx q[9],q[15];
u1(0.5017119571514081) q[15];
cx q[9],q[15];
cx q[9],q[17];
u1(0.5017119571514081) q[17];
cx q[9],q[17];
cx q[9],q[18];
u1(0.5017119571514081) q[18];
cx q[9],q[18];
cx q[9],q[19];
u1(0.5017119571514081) q[19];
cx q[9],q[19];
cx q[10],q[11];
u1(0.5017119571514081) q[11];
cx q[10],q[11];
cx q[10],q[16];
u1(0.5017119571514081) q[16];
cx q[10],q[16];
cx q[10],q[18];
u1(0.5017119571514081) q[18];
cx q[10],q[18];
cx q[10],q[19];
u1(0.5017119571514081) q[19];
cx q[10],q[19];
cx q[11],q[12];
u1(0.5017119571514081) q[12];
cx q[11],q[12];
cx q[11],q[14];
u1(0.5017119571514081) q[14];
cx q[11],q[14];
cx q[11],q[15];
u1(0.5017119571514081) q[15];
cx q[11],q[15];
cx q[11],q[16];
u1(0.5017119571514081) q[16];
cx q[11],q[16];
cx q[11],q[17];
u1(0.5017119571514081) q[17];
cx q[11],q[17];
cx q[11],q[18];
u1(0.5017119571514081) q[18];
cx q[11],q[18];
cx q[11],q[19];
u1(0.5017119571514081) q[19];
cx q[11],q[19];
cx q[12],q[14];
u1(0.5017119571514081) q[14];
cx q[12],q[14];
cx q[12],q[16];
u1(0.5017119571514081) q[16];
cx q[12],q[16];
cx q[12],q[17];
u1(0.5017119571514081) q[17];
cx q[12],q[17];
cx q[12],q[18];
u1(0.5017119571514081) q[18];
cx q[12],q[18];
cx q[13],q[14];
u1(0.5017119571514081) q[14];
cx q[13],q[14];
cx q[13],q[15];
u1(0.5017119571514081) q[15];
cx q[13],q[15];
cx q[13],q[19];
u1(0.5017119571514081) q[19];
cx q[13],q[19];
cx q[14],q[17];
u1(0.5017119571514081) q[17];
cx q[14],q[17];
cx q[14],q[19];
u1(0.5017119571514081) q[19];
cx q[14],q[19];
cx q[15],q[16];
u1(0.5017119571514081) q[16];
cx q[15],q[16];
cx q[15],q[17];
u1(0.5017119571514081) q[17];
cx q[15],q[17];
cx q[15],q[19];
u1(0.5017119571514081) q[19];
cx q[15],q[19];
cx q[17],q[19];
u1(0.5017119571514081) q[19];
cx q[17],q[19];
h q[0];
rz(1.1583693576034666) q[0];
h q[0];
h q[1];
rz(1.1583693576034666) q[1];
h q[1];
h q[2];
rz(1.1583693576034666) q[2];
h q[2];
h q[3];
rz(1.1583693576034666) q[3];
h q[3];
h q[4];
rz(1.1583693576034666) q[4];
h q[4];
h q[5];
rz(1.1583693576034666) q[5];
h q[5];
h q[6];
rz(1.1583693576034666) q[6];
h q[6];
h q[7];
rz(1.1583693576034666) q[7];
h q[7];
h q[8];
rz(1.1583693576034666) q[8];
h q[8];
h q[9];
rz(1.1583693576034666) q[9];
h q[9];
h q[10];
rz(1.1583693576034666) q[10];
h q[10];
h q[11];
rz(1.1583693576034666) q[11];
h q[11];
h q[12];
rz(1.1583693576034666) q[12];
h q[12];
h q[13];
rz(1.1583693576034666) q[13];
h q[13];
h q[14];
rz(1.1583693576034666) q[14];
h q[14];
h q[15];
rz(1.1583693576034666) q[15];
h q[15];
h q[16];
rz(1.1583693576034666) q[16];
h q[16];
h q[17];
rz(1.1583693576034666) q[17];
h q[17];
h q[18];
rz(1.1583693576034666) q[18];
h q[18];
h q[19];
rz(1.1583693576034666) q[19];
h q[19];
cx q[0],q[1];
u1(0.5393354062064886) q[1];
cx q[0],q[1];
cx q[0],q[2];
u1(0.5393354062064886) q[2];
cx q[0],q[2];
cx q[0],q[3];
u1(0.5393354062064886) q[3];
cx q[0],q[3];
cx q[0],q[5];
u1(0.5393354062064886) q[5];
cx q[0],q[5];
cx q[0],q[7];
u1(0.5393354062064886) q[7];
cx q[0],q[7];
cx q[0],q[8];
u1(0.5393354062064886) q[8];
cx q[0],q[8];
cx q[0],q[9];
u1(0.5393354062064886) q[9];
cx q[0],q[9];
cx q[0],q[10];
u1(0.5393354062064886) q[10];
cx q[0],q[10];
cx q[0],q[12];
u1(0.5393354062064886) q[12];
cx q[0],q[12];
cx q[0],q[14];
u1(0.5393354062064886) q[14];
cx q[0],q[14];
cx q[0],q[16];
u1(0.5393354062064886) q[16];
cx q[0],q[16];
cx q[0],q[17];
u1(0.5393354062064886) q[17];
cx q[0],q[17];
cx q[0],q[19];
u1(0.5393354062064886) q[19];
cx q[0],q[19];
cx q[1],q[2];
u1(0.5393354062064886) q[2];
cx q[1],q[2];
cx q[1],q[4];
u1(0.5393354062064886) q[4];
cx q[1],q[4];
cx q[1],q[7];
u1(0.5393354062064886) q[7];
cx q[1],q[7];
cx q[1],q[9];
u1(0.5393354062064886) q[9];
cx q[1],q[9];
cx q[1],q[12];
u1(0.5393354062064886) q[12];
cx q[1],q[12];
cx q[1],q[13];
u1(0.5393354062064886) q[13];
cx q[1],q[13];
cx q[1],q[14];
u1(0.5393354062064886) q[14];
cx q[1],q[14];
cx q[1],q[18];
u1(0.5393354062064886) q[18];
cx q[1],q[18];
cx q[2],q[3];
u1(0.5393354062064886) q[3];
cx q[2],q[3];
cx q[2],q[4];
u1(0.5393354062064886) q[4];
cx q[2],q[4];
cx q[2],q[8];
u1(0.5393354062064886) q[8];
cx q[2],q[8];
cx q[2],q[10];
u1(0.5393354062064886) q[10];
cx q[2],q[10];
cx q[2],q[11];
u1(0.5393354062064886) q[11];
cx q[2],q[11];
cx q[2],q[13];
u1(0.5393354062064886) q[13];
cx q[2],q[13];
cx q[2],q[14];
u1(0.5393354062064886) q[14];
cx q[2],q[14];
cx q[2],q[15];
u1(0.5393354062064886) q[15];
cx q[2],q[15];
cx q[2],q[18];
u1(0.5393354062064886) q[18];
cx q[2],q[18];
cx q[2],q[19];
u1(0.5393354062064886) q[19];
cx q[2],q[19];
cx q[3],q[4];
u1(0.5393354062064886) q[4];
cx q[3],q[4];
cx q[3],q[5];
u1(0.5393354062064886) q[5];
cx q[3],q[5];
cx q[3],q[6];
u1(0.5393354062064886) q[6];
cx q[3],q[6];
cx q[3],q[8];
u1(0.5393354062064886) q[8];
cx q[3],q[8];
cx q[3],q[9];
u1(0.5393354062064886) q[9];
cx q[3],q[9];
cx q[3],q[16];
u1(0.5393354062064886) q[16];
cx q[3],q[16];
cx q[3],q[19];
u1(0.5393354062064886) q[19];
cx q[3],q[19];
cx q[4],q[5];
u1(0.5393354062064886) q[5];
cx q[4],q[5];
cx q[4],q[6];
u1(0.5393354062064886) q[6];
cx q[4],q[6];
cx q[4],q[7];
u1(0.5393354062064886) q[7];
cx q[4],q[7];
cx q[4],q[9];
u1(0.5393354062064886) q[9];
cx q[4],q[9];
cx q[4],q[10];
u1(0.5393354062064886) q[10];
cx q[4],q[10];
cx q[4],q[14];
u1(0.5393354062064886) q[14];
cx q[4],q[14];
cx q[4],q[16];
u1(0.5393354062064886) q[16];
cx q[4],q[16];
cx q[4],q[17];
u1(0.5393354062064886) q[17];
cx q[4],q[17];
cx q[4],q[19];
u1(0.5393354062064886) q[19];
cx q[4],q[19];
cx q[5],q[6];
u1(0.5393354062064886) q[6];
cx q[5],q[6];
cx q[5],q[8];
u1(0.5393354062064886) q[8];
cx q[5],q[8];
cx q[5],q[9];
u1(0.5393354062064886) q[9];
cx q[5],q[9];
cx q[5],q[12];
u1(0.5393354062064886) q[12];
cx q[5],q[12];
cx q[5],q[13];
u1(0.5393354062064886) q[13];
cx q[5],q[13];
cx q[5],q[19];
u1(0.5393354062064886) q[19];
cx q[5],q[19];
cx q[6],q[8];
u1(0.5393354062064886) q[8];
cx q[6],q[8];
cx q[6],q[14];
u1(0.5393354062064886) q[14];
cx q[6],q[14];
cx q[6],q[15];
u1(0.5393354062064886) q[15];
cx q[6],q[15];
cx q[6],q[18];
u1(0.5393354062064886) q[18];
cx q[6],q[18];
cx q[7],q[8];
u1(0.5393354062064886) q[8];
cx q[7],q[8];
cx q[7],q[9];
u1(0.5393354062064886) q[9];
cx q[7],q[9];
cx q[7],q[10];
u1(0.5393354062064886) q[10];
cx q[7],q[10];
cx q[7],q[11];
u1(0.5393354062064886) q[11];
cx q[7],q[11];
cx q[7],q[13];
u1(0.5393354062064886) q[13];
cx q[7],q[13];
cx q[7],q[14];
u1(0.5393354062064886) q[14];
cx q[7],q[14];
cx q[7],q[18];
u1(0.5393354062064886) q[18];
cx q[7],q[18];
cx q[8],q[11];
u1(0.5393354062064886) q[11];
cx q[8],q[11];
cx q[8],q[12];
u1(0.5393354062064886) q[12];
cx q[8],q[12];
cx q[8],q[13];
u1(0.5393354062064886) q[13];
cx q[8],q[13];
cx q[8],q[15];
u1(0.5393354062064886) q[15];
cx q[8],q[15];
cx q[8],q[16];
u1(0.5393354062064886) q[16];
cx q[8],q[16];
cx q[8],q[17];
u1(0.5393354062064886) q[17];
cx q[8],q[17];
cx q[8],q[18];
u1(0.5393354062064886) q[18];
cx q[8],q[18];
cx q[9],q[11];
u1(0.5393354062064886) q[11];
cx q[9],q[11];
cx q[9],q[12];
u1(0.5393354062064886) q[12];
cx q[9],q[12];
cx q[9],q[14];
u1(0.5393354062064886) q[14];
cx q[9],q[14];
cx q[9],q[15];
u1(0.5393354062064886) q[15];
cx q[9],q[15];
cx q[9],q[17];
u1(0.5393354062064886) q[17];
cx q[9],q[17];
cx q[9],q[18];
u1(0.5393354062064886) q[18];
cx q[9],q[18];
cx q[9],q[19];
u1(0.5393354062064886) q[19];
cx q[9],q[19];
cx q[10],q[11];
u1(0.5393354062064886) q[11];
cx q[10],q[11];
cx q[10],q[16];
u1(0.5393354062064886) q[16];
cx q[10],q[16];
cx q[10],q[18];
u1(0.5393354062064886) q[18];
cx q[10],q[18];
cx q[10],q[19];
u1(0.5393354062064886) q[19];
cx q[10],q[19];
cx q[11],q[12];
u1(0.5393354062064886) q[12];
cx q[11],q[12];
cx q[11],q[14];
u1(0.5393354062064886) q[14];
cx q[11],q[14];
cx q[11],q[15];
u1(0.5393354062064886) q[15];
cx q[11],q[15];
cx q[11],q[16];
u1(0.5393354062064886) q[16];
cx q[11],q[16];
cx q[11],q[17];
u1(0.5393354062064886) q[17];
cx q[11],q[17];
cx q[11],q[18];
u1(0.5393354062064886) q[18];
cx q[11],q[18];
cx q[11],q[19];
u1(0.5393354062064886) q[19];
cx q[11],q[19];
cx q[12],q[14];
u1(0.5393354062064886) q[14];
cx q[12],q[14];
cx q[12],q[16];
u1(0.5393354062064886) q[16];
cx q[12],q[16];
cx q[12],q[17];
u1(0.5393354062064886) q[17];
cx q[12],q[17];
cx q[12],q[18];
u1(0.5393354062064886) q[18];
cx q[12],q[18];
cx q[13],q[14];
u1(0.5393354062064886) q[14];
cx q[13],q[14];
cx q[13],q[15];
u1(0.5393354062064886) q[15];
cx q[13],q[15];
cx q[13],q[19];
u1(0.5393354062064886) q[19];
cx q[13],q[19];
cx q[14],q[17];
u1(0.5393354062064886) q[17];
cx q[14],q[17];
cx q[14],q[19];
u1(0.5393354062064886) q[19];
cx q[14],q[19];
cx q[15],q[16];
u1(0.5393354062064886) q[16];
cx q[15],q[16];
cx q[15],q[17];
u1(0.5393354062064886) q[17];
cx q[15],q[17];
cx q[15],q[19];
u1(0.5393354062064886) q[19];
cx q[15],q[19];
cx q[17],q[19];
u1(0.5393354062064886) q[19];
cx q[17],q[19];
h q[0];
rz(1.582573278993123) q[0];
h q[0];
h q[1];
rz(1.582573278993123) q[1];
h q[1];
h q[2];
rz(1.582573278993123) q[2];
h q[2];
h q[3];
rz(1.582573278993123) q[3];
h q[3];
h q[4];
rz(1.582573278993123) q[4];
h q[4];
h q[5];
rz(1.582573278993123) q[5];
h q[5];
h q[6];
rz(1.582573278993123) q[6];
h q[6];
h q[7];
rz(1.582573278993123) q[7];
h q[7];
h q[8];
rz(1.582573278993123) q[8];
h q[8];
h q[9];
rz(1.582573278993123) q[9];
h q[9];
h q[10];
rz(1.582573278993123) q[10];
h q[10];
h q[11];
rz(1.582573278993123) q[11];
h q[11];
h q[12];
rz(1.582573278993123) q[12];
h q[12];
h q[13];
rz(1.582573278993123) q[13];
h q[13];
h q[14];
rz(1.582573278993123) q[14];
h q[14];
h q[15];
rz(1.582573278993123) q[15];
h q[15];
h q[16];
rz(1.582573278993123) q[16];
h q[16];
h q[17];
rz(1.582573278993123) q[17];
h q[17];
h q[18];
rz(1.582573278993123) q[18];
h q[18];
h q[19];
rz(1.582573278993123) q[19];
h q[19];
cx q[0],q[1];
u1(0.13587420195528843) q[1];
cx q[0],q[1];
cx q[0],q[2];
u1(0.13587420195528843) q[2];
cx q[0],q[2];
cx q[0],q[3];
u1(0.13587420195528843) q[3];
cx q[0],q[3];
cx q[0],q[5];
u1(0.13587420195528843) q[5];
cx q[0],q[5];
cx q[0],q[7];
u1(0.13587420195528843) q[7];
cx q[0],q[7];
cx q[0],q[8];
u1(0.13587420195528843) q[8];
cx q[0],q[8];
cx q[0],q[9];
u1(0.13587420195528843) q[9];
cx q[0],q[9];
cx q[0],q[10];
u1(0.13587420195528843) q[10];
cx q[0],q[10];
cx q[0],q[12];
u1(0.13587420195528843) q[12];
cx q[0],q[12];
cx q[0],q[14];
u1(0.13587420195528843) q[14];
cx q[0],q[14];
cx q[0],q[16];
u1(0.13587420195528843) q[16];
cx q[0],q[16];
cx q[0],q[17];
u1(0.13587420195528843) q[17];
cx q[0],q[17];
cx q[0],q[19];
u1(0.13587420195528843) q[19];
cx q[0],q[19];
cx q[1],q[2];
u1(0.13587420195528843) q[2];
cx q[1],q[2];
cx q[1],q[4];
u1(0.13587420195528843) q[4];
cx q[1],q[4];
cx q[1],q[7];
u1(0.13587420195528843) q[7];
cx q[1],q[7];
cx q[1],q[9];
u1(0.13587420195528843) q[9];
cx q[1],q[9];
cx q[1],q[12];
u1(0.13587420195528843) q[12];
cx q[1],q[12];
cx q[1],q[13];
u1(0.13587420195528843) q[13];
cx q[1],q[13];
cx q[1],q[14];
u1(0.13587420195528843) q[14];
cx q[1],q[14];
cx q[1],q[18];
u1(0.13587420195528843) q[18];
cx q[1],q[18];
cx q[2],q[3];
u1(0.13587420195528843) q[3];
cx q[2],q[3];
cx q[2],q[4];
u1(0.13587420195528843) q[4];
cx q[2],q[4];
cx q[2],q[8];
u1(0.13587420195528843) q[8];
cx q[2],q[8];
cx q[2],q[10];
u1(0.13587420195528843) q[10];
cx q[2],q[10];
cx q[2],q[11];
u1(0.13587420195528843) q[11];
cx q[2],q[11];
cx q[2],q[13];
u1(0.13587420195528843) q[13];
cx q[2],q[13];
cx q[2],q[14];
u1(0.13587420195528843) q[14];
cx q[2],q[14];
cx q[2],q[15];
u1(0.13587420195528843) q[15];
cx q[2],q[15];
cx q[2],q[18];
u1(0.13587420195528843) q[18];
cx q[2],q[18];
cx q[2],q[19];
u1(0.13587420195528843) q[19];
cx q[2],q[19];
cx q[3],q[4];
u1(0.13587420195528843) q[4];
cx q[3],q[4];
cx q[3],q[5];
u1(0.13587420195528843) q[5];
cx q[3],q[5];
cx q[3],q[6];
u1(0.13587420195528843) q[6];
cx q[3],q[6];
cx q[3],q[8];
u1(0.13587420195528843) q[8];
cx q[3],q[8];
cx q[3],q[9];
u1(0.13587420195528843) q[9];
cx q[3],q[9];
cx q[3],q[16];
u1(0.13587420195528843) q[16];
cx q[3],q[16];
cx q[3],q[19];
u1(0.13587420195528843) q[19];
cx q[3],q[19];
cx q[4],q[5];
u1(0.13587420195528843) q[5];
cx q[4],q[5];
cx q[4],q[6];
u1(0.13587420195528843) q[6];
cx q[4],q[6];
cx q[4],q[7];
u1(0.13587420195528843) q[7];
cx q[4],q[7];
cx q[4],q[9];
u1(0.13587420195528843) q[9];
cx q[4],q[9];
cx q[4],q[10];
u1(0.13587420195528843) q[10];
cx q[4],q[10];
cx q[4],q[14];
u1(0.13587420195528843) q[14];
cx q[4],q[14];
cx q[4],q[16];
u1(0.13587420195528843) q[16];
cx q[4],q[16];
cx q[4],q[17];
u1(0.13587420195528843) q[17];
cx q[4],q[17];
cx q[4],q[19];
u1(0.13587420195528843) q[19];
cx q[4],q[19];
cx q[5],q[6];
u1(0.13587420195528843) q[6];
cx q[5],q[6];
cx q[5],q[8];
u1(0.13587420195528843) q[8];
cx q[5],q[8];
cx q[5],q[9];
u1(0.13587420195528843) q[9];
cx q[5],q[9];
cx q[5],q[12];
u1(0.13587420195528843) q[12];
cx q[5],q[12];
cx q[5],q[13];
u1(0.13587420195528843) q[13];
cx q[5],q[13];
cx q[5],q[19];
u1(0.13587420195528843) q[19];
cx q[5],q[19];
cx q[6],q[8];
u1(0.13587420195528843) q[8];
cx q[6],q[8];
cx q[6],q[14];
u1(0.13587420195528843) q[14];
cx q[6],q[14];
cx q[6],q[15];
u1(0.13587420195528843) q[15];
cx q[6],q[15];
cx q[6],q[18];
u1(0.13587420195528843) q[18];
cx q[6],q[18];
cx q[7],q[8];
u1(0.13587420195528843) q[8];
cx q[7],q[8];
cx q[7],q[9];
u1(0.13587420195528843) q[9];
cx q[7],q[9];
cx q[7],q[10];
u1(0.13587420195528843) q[10];
cx q[7],q[10];
cx q[7],q[11];
u1(0.13587420195528843) q[11];
cx q[7],q[11];
cx q[7],q[13];
u1(0.13587420195528843) q[13];
cx q[7],q[13];
cx q[7],q[14];
u1(0.13587420195528843) q[14];
cx q[7],q[14];
cx q[7],q[18];
u1(0.13587420195528843) q[18];
cx q[7],q[18];
cx q[8],q[11];
u1(0.13587420195528843) q[11];
cx q[8],q[11];
cx q[8],q[12];
u1(0.13587420195528843) q[12];
cx q[8],q[12];
cx q[8],q[13];
u1(0.13587420195528843) q[13];
cx q[8],q[13];
cx q[8],q[15];
u1(0.13587420195528843) q[15];
cx q[8],q[15];
cx q[8],q[16];
u1(0.13587420195528843) q[16];
cx q[8],q[16];
cx q[8],q[17];
u1(0.13587420195528843) q[17];
cx q[8],q[17];
cx q[8],q[18];
u1(0.13587420195528843) q[18];
cx q[8],q[18];
cx q[9],q[11];
u1(0.13587420195528843) q[11];
cx q[9],q[11];
cx q[9],q[12];
u1(0.13587420195528843) q[12];
cx q[9],q[12];
cx q[9],q[14];
u1(0.13587420195528843) q[14];
cx q[9],q[14];
cx q[9],q[15];
u1(0.13587420195528843) q[15];
cx q[9],q[15];
cx q[9],q[17];
u1(0.13587420195528843) q[17];
cx q[9],q[17];
cx q[9],q[18];
u1(0.13587420195528843) q[18];
cx q[9],q[18];
cx q[9],q[19];
u1(0.13587420195528843) q[19];
cx q[9],q[19];
cx q[10],q[11];
u1(0.13587420195528843) q[11];
cx q[10],q[11];
cx q[10],q[16];
u1(0.13587420195528843) q[16];
cx q[10],q[16];
cx q[10],q[18];
u1(0.13587420195528843) q[18];
cx q[10],q[18];
cx q[10],q[19];
u1(0.13587420195528843) q[19];
cx q[10],q[19];
cx q[11],q[12];
u1(0.13587420195528843) q[12];
cx q[11],q[12];
cx q[11],q[14];
u1(0.13587420195528843) q[14];
cx q[11],q[14];
cx q[11],q[15];
u1(0.13587420195528843) q[15];
cx q[11],q[15];
cx q[11],q[16];
u1(0.13587420195528843) q[16];
cx q[11],q[16];
cx q[11],q[17];
u1(0.13587420195528843) q[17];
cx q[11],q[17];
cx q[11],q[18];
u1(0.13587420195528843) q[18];
cx q[11],q[18];
cx q[11],q[19];
u1(0.13587420195528843) q[19];
cx q[11],q[19];
cx q[12],q[14];
u1(0.13587420195528843) q[14];
cx q[12],q[14];
cx q[12],q[16];
u1(0.13587420195528843) q[16];
cx q[12],q[16];
cx q[12],q[17];
u1(0.13587420195528843) q[17];
cx q[12],q[17];
cx q[12],q[18];
u1(0.13587420195528843) q[18];
cx q[12],q[18];
cx q[13],q[14];
u1(0.13587420195528843) q[14];
cx q[13],q[14];
cx q[13],q[15];
u1(0.13587420195528843) q[15];
cx q[13],q[15];
cx q[13],q[19];
u1(0.13587420195528843) q[19];
cx q[13],q[19];
cx q[14],q[17];
u1(0.13587420195528843) q[17];
cx q[14],q[17];
cx q[14],q[19];
u1(0.13587420195528843) q[19];
cx q[14],q[19];
cx q[15],q[16];
u1(0.13587420195528843) q[16];
cx q[15],q[16];
cx q[15],q[17];
u1(0.13587420195528843) q[17];
cx q[15],q[17];
cx q[15],q[19];
u1(0.13587420195528843) q[19];
cx q[15],q[19];
cx q[17],q[19];
u1(0.13587420195528843) q[19];
cx q[17],q[19];
h q[0];
rz(0.3021479653988841) q[0];
h q[0];
h q[1];
rz(0.3021479653988841) q[1];
h q[1];
h q[2];
rz(0.3021479653988841) q[2];
h q[2];
h q[3];
rz(0.3021479653988841) q[3];
h q[3];
h q[4];
rz(0.3021479653988841) q[4];
h q[4];
h q[5];
rz(0.3021479653988841) q[5];
h q[5];
h q[6];
rz(0.3021479653988841) q[6];
h q[6];
h q[7];
rz(0.3021479653988841) q[7];
h q[7];
h q[8];
rz(0.3021479653988841) q[8];
h q[8];
h q[9];
rz(0.3021479653988841) q[9];
h q[9];
h q[10];
rz(0.3021479653988841) q[10];
h q[10];
h q[11];
rz(0.3021479653988841) q[11];
h q[11];
h q[12];
rz(0.3021479653988841) q[12];
h q[12];
h q[13];
rz(0.3021479653988841) q[13];
h q[13];
h q[14];
rz(0.3021479653988841) q[14];
h q[14];
h q[15];
rz(0.3021479653988841) q[15];
h q[15];
h q[16];
rz(0.3021479653988841) q[16];
h q[16];
h q[17];
rz(0.3021479653988841) q[17];
h q[17];
h q[18];
rz(0.3021479653988841) q[18];
h q[18];
h q[19];
rz(0.3021479653988841) q[19];
h q[19];
cx q[0],q[1];
u1(0.306386568421659) q[1];
cx q[0],q[1];
cx q[0],q[2];
u1(0.306386568421659) q[2];
cx q[0],q[2];
cx q[0],q[3];
u1(0.306386568421659) q[3];
cx q[0],q[3];
cx q[0],q[5];
u1(0.306386568421659) q[5];
cx q[0],q[5];
cx q[0],q[7];
u1(0.306386568421659) q[7];
cx q[0],q[7];
cx q[0],q[8];
u1(0.306386568421659) q[8];
cx q[0],q[8];
cx q[0],q[9];
u1(0.306386568421659) q[9];
cx q[0],q[9];
cx q[0],q[10];
u1(0.306386568421659) q[10];
cx q[0],q[10];
cx q[0],q[12];
u1(0.306386568421659) q[12];
cx q[0],q[12];
cx q[0],q[14];
u1(0.306386568421659) q[14];
cx q[0],q[14];
cx q[0],q[16];
u1(0.306386568421659) q[16];
cx q[0],q[16];
cx q[0],q[17];
u1(0.306386568421659) q[17];
cx q[0],q[17];
cx q[0],q[19];
u1(0.306386568421659) q[19];
cx q[0],q[19];
cx q[1],q[2];
u1(0.306386568421659) q[2];
cx q[1],q[2];
cx q[1],q[4];
u1(0.306386568421659) q[4];
cx q[1],q[4];
cx q[1],q[7];
u1(0.306386568421659) q[7];
cx q[1],q[7];
cx q[1],q[9];
u1(0.306386568421659) q[9];
cx q[1],q[9];
cx q[1],q[12];
u1(0.306386568421659) q[12];
cx q[1],q[12];
cx q[1],q[13];
u1(0.306386568421659) q[13];
cx q[1],q[13];
cx q[1],q[14];
u1(0.306386568421659) q[14];
cx q[1],q[14];
cx q[1],q[18];
u1(0.306386568421659) q[18];
cx q[1],q[18];
cx q[2],q[3];
u1(0.306386568421659) q[3];
cx q[2],q[3];
cx q[2],q[4];
u1(0.306386568421659) q[4];
cx q[2],q[4];
cx q[2],q[8];
u1(0.306386568421659) q[8];
cx q[2],q[8];
cx q[2],q[10];
u1(0.306386568421659) q[10];
cx q[2],q[10];
cx q[2],q[11];
u1(0.306386568421659) q[11];
cx q[2],q[11];
cx q[2],q[13];
u1(0.306386568421659) q[13];
cx q[2],q[13];
cx q[2],q[14];
u1(0.306386568421659) q[14];
cx q[2],q[14];
cx q[2],q[15];
u1(0.306386568421659) q[15];
cx q[2],q[15];
cx q[2],q[18];
u1(0.306386568421659) q[18];
cx q[2],q[18];
cx q[2],q[19];
u1(0.306386568421659) q[19];
cx q[2],q[19];
cx q[3],q[4];
u1(0.306386568421659) q[4];
cx q[3],q[4];
cx q[3],q[5];
u1(0.306386568421659) q[5];
cx q[3],q[5];
cx q[3],q[6];
u1(0.306386568421659) q[6];
cx q[3],q[6];
cx q[3],q[8];
u1(0.306386568421659) q[8];
cx q[3],q[8];
cx q[3],q[9];
u1(0.306386568421659) q[9];
cx q[3],q[9];
cx q[3],q[16];
u1(0.306386568421659) q[16];
cx q[3],q[16];
cx q[3],q[19];
u1(0.306386568421659) q[19];
cx q[3],q[19];
cx q[4],q[5];
u1(0.306386568421659) q[5];
cx q[4],q[5];
cx q[4],q[6];
u1(0.306386568421659) q[6];
cx q[4],q[6];
cx q[4],q[7];
u1(0.306386568421659) q[7];
cx q[4],q[7];
cx q[4],q[9];
u1(0.306386568421659) q[9];
cx q[4],q[9];
cx q[4],q[10];
u1(0.306386568421659) q[10];
cx q[4],q[10];
cx q[4],q[14];
u1(0.306386568421659) q[14];
cx q[4],q[14];
cx q[4],q[16];
u1(0.306386568421659) q[16];
cx q[4],q[16];
cx q[4],q[17];
u1(0.306386568421659) q[17];
cx q[4],q[17];
cx q[4],q[19];
u1(0.306386568421659) q[19];
cx q[4],q[19];
cx q[5],q[6];
u1(0.306386568421659) q[6];
cx q[5],q[6];
cx q[5],q[8];
u1(0.306386568421659) q[8];
cx q[5],q[8];
cx q[5],q[9];
u1(0.306386568421659) q[9];
cx q[5],q[9];
cx q[5],q[12];
u1(0.306386568421659) q[12];
cx q[5],q[12];
cx q[5],q[13];
u1(0.306386568421659) q[13];
cx q[5],q[13];
cx q[5],q[19];
u1(0.306386568421659) q[19];
cx q[5],q[19];
cx q[6],q[8];
u1(0.306386568421659) q[8];
cx q[6],q[8];
cx q[6],q[14];
u1(0.306386568421659) q[14];
cx q[6],q[14];
cx q[6],q[15];
u1(0.306386568421659) q[15];
cx q[6],q[15];
cx q[6],q[18];
u1(0.306386568421659) q[18];
cx q[6],q[18];
cx q[7],q[8];
u1(0.306386568421659) q[8];
cx q[7],q[8];
cx q[7],q[9];
u1(0.306386568421659) q[9];
cx q[7],q[9];
cx q[7],q[10];
u1(0.306386568421659) q[10];
cx q[7],q[10];
cx q[7],q[11];
u1(0.306386568421659) q[11];
cx q[7],q[11];
cx q[7],q[13];
u1(0.306386568421659) q[13];
cx q[7],q[13];
cx q[7],q[14];
u1(0.306386568421659) q[14];
cx q[7],q[14];
cx q[7],q[18];
u1(0.306386568421659) q[18];
cx q[7],q[18];
cx q[8],q[11];
u1(0.306386568421659) q[11];
cx q[8],q[11];
cx q[8],q[12];
u1(0.306386568421659) q[12];
cx q[8],q[12];
cx q[8],q[13];
u1(0.306386568421659) q[13];
cx q[8],q[13];
cx q[8],q[15];
u1(0.306386568421659) q[15];
cx q[8],q[15];
cx q[8],q[16];
u1(0.306386568421659) q[16];
cx q[8],q[16];
cx q[8],q[17];
u1(0.306386568421659) q[17];
cx q[8],q[17];
cx q[8],q[18];
u1(0.306386568421659) q[18];
cx q[8],q[18];
cx q[9],q[11];
u1(0.306386568421659) q[11];
cx q[9],q[11];
cx q[9],q[12];
u1(0.306386568421659) q[12];
cx q[9],q[12];
cx q[9],q[14];
u1(0.306386568421659) q[14];
cx q[9],q[14];
cx q[9],q[15];
u1(0.306386568421659) q[15];
cx q[9],q[15];
cx q[9],q[17];
u1(0.306386568421659) q[17];
cx q[9],q[17];
cx q[9],q[18];
u1(0.306386568421659) q[18];
cx q[9],q[18];
cx q[9],q[19];
u1(0.306386568421659) q[19];
cx q[9],q[19];
cx q[10],q[11];
u1(0.306386568421659) q[11];
cx q[10],q[11];
cx q[10],q[16];
u1(0.306386568421659) q[16];
cx q[10],q[16];
cx q[10],q[18];
u1(0.306386568421659) q[18];
cx q[10],q[18];
cx q[10],q[19];
u1(0.306386568421659) q[19];
cx q[10],q[19];
cx q[11],q[12];
u1(0.306386568421659) q[12];
cx q[11],q[12];
cx q[11],q[14];
u1(0.306386568421659) q[14];
cx q[11],q[14];
cx q[11],q[15];
u1(0.306386568421659) q[15];
cx q[11],q[15];
cx q[11],q[16];
u1(0.306386568421659) q[16];
cx q[11],q[16];
cx q[11],q[17];
u1(0.306386568421659) q[17];
cx q[11],q[17];
cx q[11],q[18];
u1(0.306386568421659) q[18];
cx q[11],q[18];
cx q[11],q[19];
u1(0.306386568421659) q[19];
cx q[11],q[19];
cx q[12],q[14];
u1(0.306386568421659) q[14];
cx q[12],q[14];
cx q[12],q[16];
u1(0.306386568421659) q[16];
cx q[12],q[16];
cx q[12],q[17];
u1(0.306386568421659) q[17];
cx q[12],q[17];
cx q[12],q[18];
u1(0.306386568421659) q[18];
cx q[12],q[18];
cx q[13],q[14];
u1(0.306386568421659) q[14];
cx q[13],q[14];
cx q[13],q[15];
u1(0.306386568421659) q[15];
cx q[13],q[15];
cx q[13],q[19];
u1(0.306386568421659) q[19];
cx q[13],q[19];
cx q[14],q[17];
u1(0.306386568421659) q[17];
cx q[14],q[17];
cx q[14],q[19];
u1(0.306386568421659) q[19];
cx q[14],q[19];
cx q[15],q[16];
u1(0.306386568421659) q[16];
cx q[15],q[16];
cx q[15],q[17];
u1(0.306386568421659) q[17];
cx q[15],q[17];
cx q[15],q[19];
u1(0.306386568421659) q[19];
cx q[15],q[19];
cx q[17],q[19];
u1(0.306386568421659) q[19];
cx q[17],q[19];
h q[0];
rz(0.4583277333985323) q[0];
h q[0];
h q[1];
rz(0.4583277333985323) q[1];
h q[1];
h q[2];
rz(0.4583277333985323) q[2];
h q[2];
h q[3];
rz(0.4583277333985323) q[3];
h q[3];
h q[4];
rz(0.4583277333985323) q[4];
h q[4];
h q[5];
rz(0.4583277333985323) q[5];
h q[5];
h q[6];
rz(0.4583277333985323) q[6];
h q[6];
h q[7];
rz(0.4583277333985323) q[7];
h q[7];
h q[8];
rz(0.4583277333985323) q[8];
h q[8];
h q[9];
rz(0.4583277333985323) q[9];
h q[9];
h q[10];
rz(0.4583277333985323) q[10];
h q[10];
h q[11];
rz(0.4583277333985323) q[11];
h q[11];
h q[12];
rz(0.4583277333985323) q[12];
h q[12];
h q[13];
rz(0.4583277333985323) q[13];
h q[13];
h q[14];
rz(0.4583277333985323) q[14];
h q[14];
h q[15];
rz(0.4583277333985323) q[15];
h q[15];
h q[16];
rz(0.4583277333985323) q[16];
h q[16];
h q[17];
rz(0.4583277333985323) q[17];
h q[17];
h q[18];
rz(0.4583277333985323) q[18];
h q[18];
h q[19];
rz(0.4583277333985323) q[19];
h q[19];
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
