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
cx q[0],q[23];
u1(0.567798686677545) q[23];
cx q[0],q[23];
cx q[1],q[7];
u1(0.567798686677545) q[7];
cx q[1],q[7];
cx q[1],q[21];
u1(0.567798686677545) q[21];
cx q[1],q[21];
cx q[2],q[19];
u1(0.567798686677545) q[19];
cx q[2],q[19];
cx q[3],q[18];
u1(0.567798686677545) q[18];
cx q[3],q[18];
cx q[3],q[19];
u1(0.567798686677545) q[19];
cx q[3],q[19];
cx q[3],q[23];
u1(0.567798686677545) q[23];
cx q[3],q[23];
cx q[4],q[9];
u1(0.567798686677545) q[9];
cx q[4],q[9];
cx q[4],q[16];
u1(0.567798686677545) q[16];
cx q[4],q[16];
cx q[4],q[20];
u1(0.567798686677545) q[20];
cx q[4],q[20];
cx q[4],q[21];
u1(0.567798686677545) q[21];
cx q[4],q[21];
cx q[5],q[16];
u1(0.567798686677545) q[16];
cx q[5],q[16];
cx q[6],q[7];
u1(0.567798686677545) q[7];
cx q[6],q[7];
cx q[6],q[11];
u1(0.567798686677545) q[11];
cx q[6],q[11];
cx q[6],q[16];
u1(0.567798686677545) q[16];
cx q[6],q[16];
cx q[7],q[17];
u1(0.567798686677545) q[17];
cx q[7],q[17];
cx q[8],q[19];
u1(0.567798686677545) q[19];
cx q[8],q[19];
cx q[9],q[13];
u1(0.567798686677545) q[13];
cx q[9],q[13];
cx q[11],q[14];
u1(0.567798686677545) q[14];
cx q[11],q[14];
cx q[11],q[15];
u1(0.567798686677545) q[15];
cx q[11],q[15];
cx q[11],q[21];
u1(0.567798686677545) q[21];
cx q[11],q[21];
cx q[12],q[13];
u1(0.567798686677545) q[13];
cx q[12],q[13];
cx q[12],q[15];
u1(0.567798686677545) q[15];
cx q[12],q[15];
cx q[12],q[17];
u1(0.567798686677545) q[17];
cx q[12],q[17];
cx q[13],q[15];
u1(0.567798686677545) q[15];
cx q[13],q[15];
cx q[13],q[17];
u1(0.567798686677545) q[17];
cx q[13],q[17];
cx q[13],q[18];
u1(0.567798686677545) q[18];
cx q[13],q[18];
cx q[20],q[23];
u1(0.567798686677545) q[23];
cx q[20],q[23];
h q[0];
rz(0.7735305461604745) q[0];
h q[0];
h q[1];
rz(0.7735305461604745) q[1];
h q[1];
h q[2];
rz(0.7735305461604745) q[2];
h q[2];
h q[3];
rz(0.7735305461604745) q[3];
h q[3];
h q[4];
rz(0.7735305461604745) q[4];
h q[4];
h q[5];
rz(0.7735305461604745) q[5];
h q[5];
h q[6];
rz(0.7735305461604745) q[6];
h q[6];
h q[7];
rz(0.7735305461604745) q[7];
h q[7];
h q[8];
rz(0.7735305461604745) q[8];
h q[8];
h q[9];
rz(0.7735305461604745) q[9];
h q[9];
h q[10];
rz(0.7735305461604745) q[10];
h q[10];
h q[11];
rz(0.7735305461604745) q[11];
h q[11];
h q[12];
rz(0.7735305461604745) q[12];
h q[12];
h q[13];
rz(0.7735305461604745) q[13];
h q[13];
h q[14];
rz(0.7735305461604745) q[14];
h q[14];
h q[15];
rz(0.7735305461604745) q[15];
h q[15];
h q[16];
rz(0.7735305461604745) q[16];
h q[16];
h q[17];
rz(0.7735305461604745) q[17];
h q[17];
h q[18];
rz(0.7735305461604745) q[18];
h q[18];
h q[19];
rz(0.7735305461604745) q[19];
h q[19];
h q[20];
rz(0.7735305461604745) q[20];
h q[20];
h q[21];
rz(0.7735305461604745) q[21];
h q[21];
h q[22];
rz(0.7735305461604745) q[22];
h q[22];
h q[23];
rz(0.7735305461604745) q[23];
h q[23];
cx q[0],q[23];
u1(0.07616621745704877) q[23];
cx q[0],q[23];
cx q[1],q[7];
u1(0.07616621745704877) q[7];
cx q[1],q[7];
cx q[1],q[21];
u1(0.07616621745704877) q[21];
cx q[1],q[21];
cx q[2],q[19];
u1(0.07616621745704877) q[19];
cx q[2],q[19];
cx q[3],q[18];
u1(0.07616621745704877) q[18];
cx q[3],q[18];
cx q[3],q[19];
u1(0.07616621745704877) q[19];
cx q[3],q[19];
cx q[3],q[23];
u1(0.07616621745704877) q[23];
cx q[3],q[23];
cx q[4],q[9];
u1(0.07616621745704877) q[9];
cx q[4],q[9];
cx q[4],q[16];
u1(0.07616621745704877) q[16];
cx q[4],q[16];
cx q[4],q[20];
u1(0.07616621745704877) q[20];
cx q[4],q[20];
cx q[4],q[21];
u1(0.07616621745704877) q[21];
cx q[4],q[21];
cx q[5],q[16];
u1(0.07616621745704877) q[16];
cx q[5],q[16];
cx q[6],q[7];
u1(0.07616621745704877) q[7];
cx q[6],q[7];
cx q[6],q[11];
u1(0.07616621745704877) q[11];
cx q[6],q[11];
cx q[6],q[16];
u1(0.07616621745704877) q[16];
cx q[6],q[16];
cx q[7],q[17];
u1(0.07616621745704877) q[17];
cx q[7],q[17];
cx q[8],q[19];
u1(0.07616621745704877) q[19];
cx q[8],q[19];
cx q[9],q[13];
u1(0.07616621745704877) q[13];
cx q[9],q[13];
cx q[11],q[14];
u1(0.07616621745704877) q[14];
cx q[11],q[14];
cx q[11],q[15];
u1(0.07616621745704877) q[15];
cx q[11],q[15];
cx q[11],q[21];
u1(0.07616621745704877) q[21];
cx q[11],q[21];
cx q[12],q[13];
u1(0.07616621745704877) q[13];
cx q[12],q[13];
cx q[12],q[15];
u1(0.07616621745704877) q[15];
cx q[12],q[15];
cx q[12],q[17];
u1(0.07616621745704877) q[17];
cx q[12],q[17];
cx q[13],q[15];
u1(0.07616621745704877) q[15];
cx q[13],q[15];
cx q[13],q[17];
u1(0.07616621745704877) q[17];
cx q[13],q[17];
cx q[13],q[18];
u1(0.07616621745704877) q[18];
cx q[13],q[18];
cx q[20],q[23];
u1(0.07616621745704877) q[23];
cx q[20],q[23];
h q[0];
rz(1.0162696155994475) q[0];
h q[0];
h q[1];
rz(1.0162696155994475) q[1];
h q[1];
h q[2];
rz(1.0162696155994475) q[2];
h q[2];
h q[3];
rz(1.0162696155994475) q[3];
h q[3];
h q[4];
rz(1.0162696155994475) q[4];
h q[4];
h q[5];
rz(1.0162696155994475) q[5];
h q[5];
h q[6];
rz(1.0162696155994475) q[6];
h q[6];
h q[7];
rz(1.0162696155994475) q[7];
h q[7];
h q[8];
rz(1.0162696155994475) q[8];
h q[8];
h q[9];
rz(1.0162696155994475) q[9];
h q[9];
h q[10];
rz(1.0162696155994475) q[10];
h q[10];
h q[11];
rz(1.0162696155994475) q[11];
h q[11];
h q[12];
rz(1.0162696155994475) q[12];
h q[12];
h q[13];
rz(1.0162696155994475) q[13];
h q[13];
h q[14];
rz(1.0162696155994475) q[14];
h q[14];
h q[15];
rz(1.0162696155994475) q[15];
h q[15];
h q[16];
rz(1.0162696155994475) q[16];
h q[16];
h q[17];
rz(1.0162696155994475) q[17];
h q[17];
h q[18];
rz(1.0162696155994475) q[18];
h q[18];
h q[19];
rz(1.0162696155994475) q[19];
h q[19];
h q[20];
rz(1.0162696155994475) q[20];
h q[20];
h q[21];
rz(1.0162696155994475) q[21];
h q[21];
h q[22];
rz(1.0162696155994475) q[22];
h q[22];
h q[23];
rz(1.0162696155994475) q[23];
h q[23];
cx q[0],q[23];
u1(0.7744832375550238) q[23];
cx q[0],q[23];
cx q[1],q[7];
u1(0.7744832375550238) q[7];
cx q[1],q[7];
cx q[1],q[21];
u1(0.7744832375550238) q[21];
cx q[1],q[21];
cx q[2],q[19];
u1(0.7744832375550238) q[19];
cx q[2],q[19];
cx q[3],q[18];
u1(0.7744832375550238) q[18];
cx q[3],q[18];
cx q[3],q[19];
u1(0.7744832375550238) q[19];
cx q[3],q[19];
cx q[3],q[23];
u1(0.7744832375550238) q[23];
cx q[3],q[23];
cx q[4],q[9];
u1(0.7744832375550238) q[9];
cx q[4],q[9];
cx q[4],q[16];
u1(0.7744832375550238) q[16];
cx q[4],q[16];
cx q[4],q[20];
u1(0.7744832375550238) q[20];
cx q[4],q[20];
cx q[4],q[21];
u1(0.7744832375550238) q[21];
cx q[4],q[21];
cx q[5],q[16];
u1(0.7744832375550238) q[16];
cx q[5],q[16];
cx q[6],q[7];
u1(0.7744832375550238) q[7];
cx q[6],q[7];
cx q[6],q[11];
u1(0.7744832375550238) q[11];
cx q[6],q[11];
cx q[6],q[16];
u1(0.7744832375550238) q[16];
cx q[6],q[16];
cx q[7],q[17];
u1(0.7744832375550238) q[17];
cx q[7],q[17];
cx q[8],q[19];
u1(0.7744832375550238) q[19];
cx q[8],q[19];
cx q[9],q[13];
u1(0.7744832375550238) q[13];
cx q[9],q[13];
cx q[11],q[14];
u1(0.7744832375550238) q[14];
cx q[11],q[14];
cx q[11],q[15];
u1(0.7744832375550238) q[15];
cx q[11],q[15];
cx q[11],q[21];
u1(0.7744832375550238) q[21];
cx q[11],q[21];
cx q[12],q[13];
u1(0.7744832375550238) q[13];
cx q[12],q[13];
cx q[12],q[15];
u1(0.7744832375550238) q[15];
cx q[12],q[15];
cx q[12],q[17];
u1(0.7744832375550238) q[17];
cx q[12],q[17];
cx q[13],q[15];
u1(0.7744832375550238) q[15];
cx q[13],q[15];
cx q[13],q[17];
u1(0.7744832375550238) q[17];
cx q[13],q[17];
cx q[13],q[18];
u1(0.7744832375550238) q[18];
cx q[13],q[18];
cx q[20],q[23];
u1(0.7744832375550238) q[23];
cx q[20],q[23];
h q[0];
rz(0.1322621604109142) q[0];
h q[0];
h q[1];
rz(0.1322621604109142) q[1];
h q[1];
h q[2];
rz(0.1322621604109142) q[2];
h q[2];
h q[3];
rz(0.1322621604109142) q[3];
h q[3];
h q[4];
rz(0.1322621604109142) q[4];
h q[4];
h q[5];
rz(0.1322621604109142) q[5];
h q[5];
h q[6];
rz(0.1322621604109142) q[6];
h q[6];
h q[7];
rz(0.1322621604109142) q[7];
h q[7];
h q[8];
rz(0.1322621604109142) q[8];
h q[8];
h q[9];
rz(0.1322621604109142) q[9];
h q[9];
h q[10];
rz(0.1322621604109142) q[10];
h q[10];
h q[11];
rz(0.1322621604109142) q[11];
h q[11];
h q[12];
rz(0.1322621604109142) q[12];
h q[12];
h q[13];
rz(0.1322621604109142) q[13];
h q[13];
h q[14];
rz(0.1322621604109142) q[14];
h q[14];
h q[15];
rz(0.1322621604109142) q[15];
h q[15];
h q[16];
rz(0.1322621604109142) q[16];
h q[16];
h q[17];
rz(0.1322621604109142) q[17];
h q[17];
h q[18];
rz(0.1322621604109142) q[18];
h q[18];
h q[19];
rz(0.1322621604109142) q[19];
h q[19];
h q[20];
rz(0.1322621604109142) q[20];
h q[20];
h q[21];
rz(0.1322621604109142) q[21];
h q[21];
h q[22];
rz(0.1322621604109142) q[22];
h q[22];
h q[23];
rz(0.1322621604109142) q[23];
h q[23];
cx q[0],q[23];
u1(0.11127490317316424) q[23];
cx q[0],q[23];
cx q[1],q[7];
u1(0.11127490317316424) q[7];
cx q[1],q[7];
cx q[1],q[21];
u1(0.11127490317316424) q[21];
cx q[1],q[21];
cx q[2],q[19];
u1(0.11127490317316424) q[19];
cx q[2],q[19];
cx q[3],q[18];
u1(0.11127490317316424) q[18];
cx q[3],q[18];
cx q[3],q[19];
u1(0.11127490317316424) q[19];
cx q[3],q[19];
cx q[3],q[23];
u1(0.11127490317316424) q[23];
cx q[3],q[23];
cx q[4],q[9];
u1(0.11127490317316424) q[9];
cx q[4],q[9];
cx q[4],q[16];
u1(0.11127490317316424) q[16];
cx q[4],q[16];
cx q[4],q[20];
u1(0.11127490317316424) q[20];
cx q[4],q[20];
cx q[4],q[21];
u1(0.11127490317316424) q[21];
cx q[4],q[21];
cx q[5],q[16];
u1(0.11127490317316424) q[16];
cx q[5],q[16];
cx q[6],q[7];
u1(0.11127490317316424) q[7];
cx q[6],q[7];
cx q[6],q[11];
u1(0.11127490317316424) q[11];
cx q[6],q[11];
cx q[6],q[16];
u1(0.11127490317316424) q[16];
cx q[6],q[16];
cx q[7],q[17];
u1(0.11127490317316424) q[17];
cx q[7],q[17];
cx q[8],q[19];
u1(0.11127490317316424) q[19];
cx q[8],q[19];
cx q[9],q[13];
u1(0.11127490317316424) q[13];
cx q[9],q[13];
cx q[11],q[14];
u1(0.11127490317316424) q[14];
cx q[11],q[14];
cx q[11],q[15];
u1(0.11127490317316424) q[15];
cx q[11],q[15];
cx q[11],q[21];
u1(0.11127490317316424) q[21];
cx q[11],q[21];
cx q[12],q[13];
u1(0.11127490317316424) q[13];
cx q[12],q[13];
cx q[12],q[15];
u1(0.11127490317316424) q[15];
cx q[12],q[15];
cx q[12],q[17];
u1(0.11127490317316424) q[17];
cx q[12],q[17];
cx q[13],q[15];
u1(0.11127490317316424) q[15];
cx q[13],q[15];
cx q[13],q[17];
u1(0.11127490317316424) q[17];
cx q[13],q[17];
cx q[13],q[18];
u1(0.11127490317316424) q[18];
cx q[13],q[18];
cx q[20],q[23];
u1(0.11127490317316424) q[23];
cx q[20],q[23];
h q[0];
rz(0.2296615962117563) q[0];
h q[0];
h q[1];
rz(0.2296615962117563) q[1];
h q[1];
h q[2];
rz(0.2296615962117563) q[2];
h q[2];
h q[3];
rz(0.2296615962117563) q[3];
h q[3];
h q[4];
rz(0.2296615962117563) q[4];
h q[4];
h q[5];
rz(0.2296615962117563) q[5];
h q[5];
h q[6];
rz(0.2296615962117563) q[6];
h q[6];
h q[7];
rz(0.2296615962117563) q[7];
h q[7];
h q[8];
rz(0.2296615962117563) q[8];
h q[8];
h q[9];
rz(0.2296615962117563) q[9];
h q[9];
h q[10];
rz(0.2296615962117563) q[10];
h q[10];
h q[11];
rz(0.2296615962117563) q[11];
h q[11];
h q[12];
rz(0.2296615962117563) q[12];
h q[12];
h q[13];
rz(0.2296615962117563) q[13];
h q[13];
h q[14];
rz(0.2296615962117563) q[14];
h q[14];
h q[15];
rz(0.2296615962117563) q[15];
h q[15];
h q[16];
rz(0.2296615962117563) q[16];
h q[16];
h q[17];
rz(0.2296615962117563) q[17];
h q[17];
h q[18];
rz(0.2296615962117563) q[18];
h q[18];
h q[19];
rz(0.2296615962117563) q[19];
h q[19];
h q[20];
rz(0.2296615962117563) q[20];
h q[20];
h q[21];
rz(0.2296615962117563) q[21];
h q[21];
h q[22];
rz(0.2296615962117563) q[22];
h q[22];
h q[23];
rz(0.2296615962117563) q[23];
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
