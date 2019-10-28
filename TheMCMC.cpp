#include <iostream>
#include <string>
#include <random>
#include <numeric>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <functional>
#include <cmath>
#include <tuple>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>

using namespace std;
using namespace Eigen;

Matrix2d SqrtSig(Matrix2d Sig) {
	if (Sig(0, 0) < 0.0) {
		cout << "Sig(0, 0) is negative" << endl;
		exit(1);
	}
	if (Sig(1, 1) < 0.0) {
		cout << "Sig(1, 1) is negative" << endl;
		exit(1);
	}
	Matrix2d RtSig;
	RtSig(0, 0) = sqrt(Sig(0, 0));
	RtSig(0, 1) = 0.0;
	RtSig(1, 0) = Sig(1, 0) / sqrt(Sig(0, 0));
	RtSig(1, 1) = sqrt(Sig(1, 1) - pow(Sig(1, 0) / sqrt(Sig(0, 0)),2));
	return RtSig;
}

double musim(double kappa, double theta, double sigmav, double rho, vector<double>& Y, vector<double>& V, vector<double>& Jy, vector<double>& Jv)
{
	const int T = 5040;
	double C[T];
	double D[T];
	double S = 0.0;
	double W = 1.0;
	double Delta = 1.0;
	for (int t = 0;t < T;t++) {
		C[t] = Y.at(t+1) - Y.at(t) - Jy.at(t);
		D[t] = V.at(t + 1) - V.at(t) - kappa * (theta - V.at(t)) * Delta - Jv.at(t);
		W += Delta / ((1.0 - pow(rho, 2)) * V.at(t));
		S += 1 / ((1.0 - pow(rho, 2)) * V.at(t)) * (C[t] - (rho / sigmav) * D[t]);
	}
	default_random_engine rand;
	normal_distribution<double> Norm(S / W, sqrt(1 / W));
	double mu = Norm(rand);
	return mu;
}

double thetasim(double mu, double kappa, double sigmav, double rho, vector<double>& Y, vector<double>& V, vector<double>& Jy, vector<double>& Jv)
{
	const int T = 5040;
	double C[T];
	double D[T];
	double S = 0.0;
	double W = 1.0;
	double Delta = 1.0;
	double theta;
	for (int t = 0;t < T;t++) {
		C[t] = Y.at(t + 1) - Y.at(t) - mu * Delta - Jy.at(t);
		D[t] = V.at(t + 1) - V.at(t) - kappa * Delta * V.at(t) - Jv.at(t);
		W += pow(kappa, 2) * Delta / ((1.0 - pow(rho, 2)) * pow(sigmav, 2) * V.at(t));
		S += kappa / ((1.0 - pow(rho, 2)) * sigmav * V.at(t)) * (D[t] / sigmav - rho * C[t]);
	}
	default_random_engine rand;
	normal_distribution<double> Norm(S / W, sqrt(1 / W));
	do {
		theta = Norm(rand);
	} while (theta < 0);
	return theta;
}

double kappasim(double mu, double theta, double sigmav, double rho, vector<double>& Y, vector<double>& V, vector<double>& Jy, vector<double>& Jv)
{
	const int T = 5040;
	double C[T];
	double D[T];
	double S = 0.0;
	double W = 1.0;
	double Delta = 1.0;
	double kappa;
	for (int t = 0;t < T;t++) {
		C[t] = Y.at(t + 1) - Y.at(t) - mu * Delta - Jy.at(t);
		D[t] = V.at(t + 1) - V.at(t) - Jv.at(t);
		W += Delta / ((1.0 - pow(rho, 2)) * pow(sigmav, 2) * V.at(t)) * pow((theta - V.at(t)), 2);
		S += 1 / ((1.0 - pow(rho, 2)) * sigmav * V.at(t)) * (theta - V.at(t)) * (D[t] / sigmav - rho * C[t]);
	}
	default_random_engine rand;
	normal_distribution<double> Norm(S / W, sqrt(1 / W));
	do {
		kappa = Norm(rand);
	} while (kappa < 0);
	return kappa;
}

Vector2d sigmavrhosim(double mu, double kappa, double theta, vector<double>& Y, vector<double>& V, vector<double>& Jy, vector<double>& Jv) {
	const int T = 5040;
	double C[T];
	double D[T];
	double S = 0.0;
	double W = 2.0;
	double D2 = 0.0;
	double Delta = 1.0;
	for (int t = 0;t < T;t++) {
		C[t] = (Y.at(t + 1) - Y.at(t) - mu * Delta - Jy.at(t)) / sqrt(V.at(t) * Delta);
		D[t] = (V.at(t + 1) - V.at(t) - kappa * (theta - V.at(t)) * Delta - Jv.at(t)) / sqrt(V.at(t) * Delta);
		W += C[t] * C[t];
		S += C[t] * D[t];
		D2 += D[t] * D[t];
	}
	default_random_engine rand;
	gamma_distribution<double> Ga(T / 2 + 2.0, pow((0.5 * D2 + 200.0 - pow(S, 2) / (2 * W)), -1));
	double omega2inv = Ga(rand);
	double omega2 = pow(omega2inv, -1);
	normal_distribution<double> Norm(S / W, sqrt(omega2 / W));
	double phi = Norm(rand);
	Vector2d sigvrho(sqrt(pow(phi,2) + omega2), phi / (sqrt(pow(phi, 2) + omega2)));
	return sigvrho;
}

double phivsim(double mu, double kappa, double theta, double omega2, vector<double>& Y, vector<double>& V, vector<double>& Jy, vector<double>& Jv) {
	const int T = 5040;
	double C[T];
	double D[T];
	double S = 0.0;
	double W = 1.0;
	double Delta = 1.0;
	for (int t = 0;t < T;t++) {
		C[t] = (Y.at(t + 1) - Y.at(t) - mu * Delta - Jy.at(t)) / sqrt(V.at(t) * Delta);
		D[t] = (V.at(t + 1) - V.at(t) - kappa * (theta - V.at(t)) * Delta - Jv.at(t)) / sqrt(V.at(t) * Delta);
		W += pow(C[t],2) / (omega2 * V[t] * Delta);
		S += C[t] * D[t] / (omega2 * V[t] * Delta);
	}
	default_random_engine rand;
	normal_distribution<double> Norm(S / W, sqrt(1.0 / W));
	double phi = Norm(rand);
	return phi;
}

double omega2sim(double mu, double kappa, double theta, double phi, vector<double>& Y, vector<double>& V, vector<double>& Jy, vector<double>& Jv) {
	const int T = 5040;
	double C[T];
	double D[T];
	double W = 200.0;
	double Delta = 1.0;
	for (int t = 0;t < T;t++) {
		C[t] = (Y.at(t + 1) - Y.at(t) - mu * Delta - Jy.at(t)) / sqrt(V.at(t) * Delta);
		D[t] = (V.at(t + 1) - V.at(t) - kappa * (theta - V.at(t)) * Delta - Jv.at(t)) / sqrt(V.at(t) * Delta);
		W += 0.5 * pow((phi * C[t] - D[t]),2) / (V[t] * Delta);
	}
	default_random_engine rand;
	normal_distribution<double> Ga(T / 2 + 2.0, pow(W, -1));
	double omega2inv = Ga(rand);
	return pow(omega2inv,-1);
}

Vector2d etasim(vector<double>& xi, double curralpha, double pribeta, double prialpha) {
	const int T = 5040;
	double xiplus = 0.0;
	double ximinus = 0.0;
	for (int t = 0;t < T;t++) {
		if (xi[t] > 0) {
			xiplus += xi.at(t);
		}
		else {
			ximinus += xi.at(t);
		}
	}
	default_random_engine rand;
	gamma_distribution<double> Ga(T + pribeta, pow((1.0 + curralpha * xiplus - ximinus / curralpha), -1));
	double beta = Ga(rand);
	gamma_distribution<double> Gam(T + prialpha, pow((1.0 + beta * xiplus), -1));
	uniform_real_distribution<double> Unif(0.0, 1.0);
	double star = Gam(rand);
	double u = Unif(rand);
	double lstar = (pow(pow(star, 2) + 1, -(T))) * exp(beta / star * ximinus);
	double lcurr = (pow(pow(curralpha, 2) + 1, -(T))) * exp(beta / curralpha * ximinus);
	double alpha;
	if (lstar / lcurr > u) {
		alpha = star;
	}
	else {
		alpha = curralpha;
	}
	Vector2d etas(beta * alpha, beta / alpha);
	return etas;
}

Vector2d mucsim(Matrix2d& nuc, vector<Vector2d>& xic, vector<double>& Bt) {
	const int T = 5040;
	double xiysum = 0.0;
	double xivsum = 0.0;
	double B = 0.0;
	for (int t = 0;t < T;t++) {
		xiysum += xic[t][0] / Bt[t];
		xivsum += xic[t][1] / Bt[t];
		B += Bt.at(t);
	}
	Vector2d xi;
	xi(0) = xiysum / T;
	xi(1) = xivsum / T;
	Matrix2d Wc;
	Matrix2d S;
	S(0, 0) = 1.0;
	S(0, 1) = 0.0;
	S(1, 0) = 0.0;
	S(1, 1) = 1.0;
	Wc = B * nuc.inverse() + S.inverse();
	Vector2d M;
	M(0) = 0.0;
	M(1) = 0.0;
	Vector2d Sc;
	Sc = B * nuc.inverse() * xi + S.inverse() * M;
	Vector2d Mu = Wc.inverse() * Sc;
	Matrix2d Sig = Wc.inverse();
	Matrix2d RtSig = SqrtSig(Sig);
	default_random_engine rand;
	normal_distribution<double> Norm(0.0, 1.0);
	Vector2d Z;
	Z(0) = Norm(rand);
	Z(1) = Norm(rand);
	Vector2d Res;
	Res = RtSig * Z + Mu;
	return Res;
}

Matrix2d nucsim(Vector2d& muc, vector<Vector2d>& xic, vector<double>& Bt) {
	const int T = 5040;
	double xiyadj = 0.0;
	double xivadj = 0.0;
	double xiyvadj = 0.0;
	for (int t = 0;t < T;t++) {
		xiyadj += pow((xic[t][0] - muc(0) * Bt.at(t)) / sqrt(Bt.at(t)),2);
		xivadj += pow((xic[t][1] - muc(1) * Bt.at(t)) / sqrt(Bt.at(t)),2);
		xiyvadj += (xic[t][0] - muc(0) * Bt.at(t)) * (xic[t][1] - muc(1) * Bt.at(t)) / Bt[t];
	}
	Matrix2d xi;
	xi(0, 0) = xiyadj + 9.0;
	xi(0, 1) = xiyvadj - 1.5; 
	xi(1, 0) = xiyvadj - 1.5; 
	xi(1, 1) = xivadj + 1.0;
	Matrix2d xirt = SqrtSig(xi.inverse());
	default_random_engine rand;
	normal_distribution<double> Norm(0.0, 1.0);
	Vector2d Z;
	Vector2d tmp;
	Matrix2d W;
	W(0, 0) = 0.0;
	W(0, 1) = 0.0;
	W(1, 0) = 0.0;
	W(1, 1) = 0.0;
	for (int t = 1;t <= T + 2;t++) {
		Z(0) = Norm(rand);
		Z(1) = Norm(rand);
		tmp = xirt * Z;
		W += tmp * tmp.transpose();
	}
	return W.inverse();
}

Vector4d lambdasim(vector<Vector3i>& N) {
	int Nysum = 0;
	int Nvsum = 0;
	int Ncsum = 0;
	const int T = 5040;
	for (int t = 0;t < T;t++) {
		Nysum += N[t][0];
		Nvsum += N[t][1];
		Ncsum += N[t][2];
	}
	double Dir[4];
	default_random_engine rand;
	gamma_distribution<double> Diry(Nysum + 2.0, 1.0);
	gamma_distribution<double> Dirv(Nvsum + 2.0, 1.0);
	gamma_distribution<double> Dirc(Ncsum + 2.0, 1.0);
	gamma_distribution<double> Dir0(T - Nysum - Nvsum - Ncsum + 40.0, 1.0);
	Dir[0] = Diry(rand);
	Dir[1] = Dirv(rand);
	Dir[2] = Dirc(rand);
	Dir[3] = Dir0(rand);
	Vector4d lambda;
	lambda(0) = Dir[0] / (Dir[0] + Dir[1] + Dir[2] + Dir[3]);
	lambda(1) = Dir[1] / (Dir[0] + Dir[1] + Dir[2] + Dir[3]);
	lambda(2) = Dir[2] / (Dir[0] + Dir[1] + Dir[2] + Dir[3]);
	lambda(3) = Dir[3] / (Dir[0] + Dir[1] + Dir[2] + Dir[3]);
	return lambda;
}

double xisim(double mu, double kappa, double theta, double sigmav, double rho, double etapl, double etamn, double Y[2], double V[2], int N) {
	default_random_engine rand;
	uniform_real_distribution<double> Unif(0.0, 1.0);
	double star;
	double u = Unif(rand);
	if (N == 1) {
		double rho2min = 1 - pow(rho, 2);
		double Delta = 1;
		double Spl = (Y[1] - Y[0] - mu * Delta - rho / sigmav * (V[1] - V[0] - kappa * (theta - V[0]) * Delta)) / (V[0] * Delta * (rho2min)) - etapl / 2.0;
		double Smn = (Y[1] - Y[0] - mu * Delta - rho / sigmav * (V[1] - V[0] - kappa * (theta - V[0]) * Delta)) / (V[0] * Delta * (rho2min)) + etamn / 2.0;
		double W = 1 / (V[0] * Delta * (rho2min));
		double Phipl = erf(Spl / sqrt(W)) / 2.0 + 0.5;
		double Phimin = erfc(Smn / sqrt(W)) / 2.0;
		double p = Phimin / (Phimin + Phipl);
		normal_distribution<double> Norm(0.0, sqrt(1 / W));
		if (u < p) {
			do {
				star = Norm(rand) + Spl / W;
			} while (star < 0);
		}
		else {
			do {
				star = Norm(rand) + Smn / W;
			} while (star >= 0);
		}
	}
	else {
		double p = etamn / (etamn + etapl);
		if (u < p) {
			exponential_distribution<double> Exp(etapl);
			star = Exp(rand);
		}
		else {
			exponential_distribution<double> Exp(etamn);
			star = -1.0 * Exp(rand);
		}
	}
	return star;
}

Vector2d xicsim(double mu, double kappa, double theta, double sigmav, double rho, Vector2d& muc, Matrix2d& nuc, double Y[2], double V[2], int N, double B) {
	Matrix2d Sig;
	Sig(0, 0) = 1.0;
	Sig(0, 1) = rho * sigmav;
	Sig(1, 0) = rho * sigmav;
	Sig(1, 1) = pow(sigmav, 2);
	Matrix2d W;
	W = N / V[0] * Sig.inverse() + 1 / B * nuc.inverse();
	Vector2d S;
	double Delta = 1;
	Vector2d muin;
	muin(0) = Y[1] - Y[0] - mu * Delta;
	muin(1) = V[1] - V[0] - kappa * (theta - V[0]) * Delta;
	S = N / V[0] * Sig.inverse() * muin + nuc.inverse() * muc;
	Vector2d Mu = W.inverse() * S;
	Matrix2d RtSig = SqrtSig(W.inverse());
	default_random_engine rand;
	normal_distribution<double> Norm(0.0, 1.0);
	Vector2d Z;
	Z(0) = Norm(rand);
	Z(1) = Norm(rand);
	return RtSig * Z + Mu;
}

Vector3i Nsim(double mu, double kappa, double theta, double sigmav, double rho, double Y[2], double V[2], double xiy, double xiv, Vector2d& xic, Vector4d& lambda) {
	double Dir[4];
	double rho2min = 1 - pow(rho, 2);
	double Delta = 1.0;
	double A1 = Y[1] - Y[0] - mu * Delta - xiy;
	double A2 = Y[1] - Y[0] - mu * Delta - xic[0];
	double A3 = Y[1] - Y[0] - mu * Delta;
	double B1 = V[1] - V[0] - kappa * (theta - V[0]) * Delta - xiv;
	double B2 = V[1] - V[0] - kappa * (theta - V[0]) * Delta - xic[1];
	double B3 = V[1] - V[0] - kappa * (theta - V[0]) * Delta;
	Dir[0] = exp(-1 / (2 * rho2min) * (pow(A1, 2) - 2 * rho * A1 * B3 + pow(B3, 2))) * lambda[0];
	Dir[1] = exp(-1 / (2 * rho2min) * (pow(A3, 2) - 2 * rho * A3 * B1 + pow(B1, 2))) * lambda[1];
	Dir[2] = exp(-1 / (2 * rho2min) * (pow(A2, 2) - 2 * rho * A2 * B2 + pow(B2, 2))) * lambda[2];
	Dir[3] = exp(-1 / (2 * rho2min) * (pow(A3, 2) - 2 * rho * A3 * B2 + pow(B3, 2))) * lambda[3];
	default_random_engine rand;
	uniform_real_distribution<double> Unif(0.0, 1.0);
	double u = Unif(rand);
	double Dirsum = Dir[0] + Dir[1] + Dir[2] + Dir[3];
	Vector3i N;
	if (u < Dir[0] / Dirsum) {
		N(0) = 1;
		N(1) = 0;
		N(2) = 0;
	}
	else if (u < (Dir[0] + Dir[1]) / Dirsum) {
		N(0) = 0;
		N(1) = 1;
		N(2) = 0;
	}
	else if (u < (Dir[0] + Dir[1] + Dir[2]) / Dirsum) {
		N(0) = 0;
		N(1) = 0;
		N(2) = 1;
	}
	else {
		N(0) = 0;
		N(1) = 0;
		N(2) = 0;
	}
	return N;
}

double Vsim(double mu, double kappa, double theta, double sigmav, double rho, double Y[2], double V[2], double Jy, double Jv, int t) {
	default_random_engine rand;
	uniform_real_distribution<double> Unif(0.0, 1.0);
	const int T = 5040;
	double Delta = 1;
	double Vnew;
	double star;
	double rho2min = 1 - pow(rho, 2);
	normal_distribution<double> Norm(0.0, 1.0);
	if (t == 0) {
		student_t_distribution<double> tdist(6.0);
		do {
			star = tdist(rand);
		} while (star < 0);
		double epsycurr = (Y[1] - Y[0] - mu * Delta - Jy) / sqrt(V[0] * Delta);
		double epsvcurr = (V[1] - V[0] - kappa * (theta - V[0]) * Delta - Jv) / sqrt(sigmav * V[0] * Delta);
		double epsystar = (Y[1] - Y[0] - mu * Delta - Jy) / sqrt(star * Delta);
		double epsvstar = (V[1] - star - kappa * (theta - star) * Delta - Jv) / sqrt(sigmav * star * Delta);
		double lcurr = 1 / V[0] * exp(-1 / rho2min * (pow(epsycurr, 2) - 2 * rho * epsycurr * epsvcurr + pow(epsycurr, 2))) / pow(1 + pow(V[0], 2) / 2.0, -7.0 / 2.0);
		double lstar = 1 / star * exp(-1 / rho2min * (pow(epsystar, 2) - 2 * rho * epsystar * epsvstar + pow(epsystar, 2))) / pow(1 + pow(star, 2) / 2.0, -7.0 / 2.0);
		double u = Unif(rand);
		if (lstar / lcurr > u) {
			Vnew = star;
		}
		else {
			Vnew = V[0];
		}
	}
	else if (t == T) {
		do {
			Vnew = sqrt(pow(sigmav, 2) * V[0] * rho2min) * Norm(rand) + V[0] + kappa * (theta - V[0]) * Delta + Jv + rho * sigmav * (Y[1] - Y[0] - mu * Delta - Jy);
		} while (Vnew < 0);
	}
	else {
		do {
			star = sqrt(pow(sigmav, 2) * V[0] * rho2min) * Norm(rand) + V[0] + kappa * (theta - V[0]) * Delta + Jv + rho * sigmav * (Y[1] - Y[0] - mu * Delta - Jy);
		} while (star < 0);
		double epsycurr = (Y[1] - Y[0] - mu * Delta - Jy) / sqrt(V[0] * Delta);
		double epsvcurr = (V[1] - V[0] - kappa * (theta - V[0]) * Delta - Jv) / sqrt(sigmav * V[0] * Delta);
		double epsystar = (Y[1] - Y[0] - mu * Delta - Jy) / sqrt(star * Delta);
		double epsvstar = (V[1] - star - kappa * (theta - star) * Delta - Jv) / sqrt(sigmav * star * Delta);
		double lcurr = 1 / V[0] * exp(-1 / rho2min * (pow(epsycurr, 2) - 2 * rho * epsycurr * epsvcurr + pow(epsycurr, 2)));
		double lstar = 1 / star * exp(-1 / rho2min * (pow(epsystar, 2) - 2 * rho * epsystar * epsvstar + pow(epsystar, 2)));
		double u = Unif(rand);
		if (lstar / lcurr > u) {
			Vnew = star;
		}
		else {
			Vnew = V[1];
		}
	}
	return Vnew;
}

double Bsim(Vector2d& xic, Vector2d& muc, Matrix2d& nuc, double B) {
	default_random_engine rand;
	double rhomin2 = nuc.determinant() / (nuc(0,0) * nuc(1,1));
	double theta = 1 / (2 * rhomin2) * (pow(muc(0), 2) / nuc(0, 0) + pow(muc(1), 2) / nuc(1, 1) - 2 * nuc(0,1) * muc(0) * muc(1) / (nuc(0,0) * nuc(1,1))) + 1.0;
	exponential_distribution<double> Exp(theta);
	uniform_real_distribution<double> Unif(0.0, 1.0);
	double star = Exp(rand);
	double Bnew;
	double lcurr = 1 / B * exp(-1 / (2 * rhomin2) * (pow(xic(0),2) / (nuc(0, 0) * B) + pow(xic(1), 2) / (nuc(1, 1) * B) - 2 * nuc(0, 1) * xic(0) * xic(1) / (nuc(0, 0) * nuc(1, 1) * B)));
	double lstar = 1 / star * exp(-1 / (2 * rhomin2) * (pow(xic(0), 2) / (nuc(0, 0) * star) + pow(xic(1), 2) / (nuc(1, 1) * star) - 2 * nuc(0, 1) * xic(0) * xic(1) / (nuc(0, 0) * nuc(1, 1) * star)));
	double u = Unif(rand);
	if (lstar / lcurr > u) {
		Bnew = star;
	}
	else {
		Bnew = B;
	}
	return Bnew;
}

int main()
{
	vector<double> Y;
	ifstream infile;
	infile.open("returns.txt");
	if (infile.fail()) {
		cout << "Unable to open file" << endl;
		exit(1); // terminate with error
	}
	double num;
	while (infile >> num) {
		Y.push_back(num);
	}
	const int T = Y.size() - 1;
	const int sims = 5000;
	const int burnin = 1000;
	const int M = sims + burnin;
	double Delta = 1.0;
	vector<double> mu;
	vector<double> theta;
	vector<double> kappa;
	vector<double> sigmav;
	vector<double> rho;
	//Vector2d sigvrhosim;
	double omega2;
	double phi;
	vector<double> etayplus;
	vector<double> etayminus;
	vector<double> etavplus;
	vector<double> etavminus;
	Vector2d etassim;
	vector<Vector2d> muc;
	vector<Matrix2d> nuc;
	vector<Vector4d> lambda;
	vector<vector<double>> xiy;
	vector<vector<double>> xiv;
	vector<vector<Vector2d>> xic;
	vector<vector<Vector3i>> N;
	vector<vector<double>> V;
	vector<vector<double>> B;
	Vector2d mucstart(-0.02, 0.01);
	Matrix2d nucstart;
	nucstart(0, 0) = 0.000225;
	nucstart(0, 1) = 0.0;
	nucstart(1, 0) = 0.0;
	nucstart(1, 1) = 0.0001;
	Vector4d lambstart(0.02, 0.02, 0.02, 0.94);
	mu.push_back(0.0005);
	kappa.push_back(0.02);
	theta.push_back(0.0001);
	sigmav.push_back(0.1);
	rho.push_back(-0.4);
	omega2 = pow(sigmav.back(), 2) * (1 - pow(rho.back(), 2));
	etayplus.push_back(25.0);
	etayminus.push_back(20.0);
	etavplus.push_back(100);
	etavminus.push_back(400);
	muc.push_back(mucstart);
	nuc.push_back(nucstart);
	lambda.push_back(lambstart);
	Vector2d ZeroF(0.0, 0.0);
	Vector3i ZeroI(0, 0, 0);
	vector<double> xiycurr(T);
	vector<double> xivcurr(T);
	vector<Vector2d> xiccurr(T);
	vector<Vector3i> Ncurr(T);
	vector<double> Vcurr(T + 1);
	vector<double> Bcurr(T);
	vector<double> Jy(T);
	vector<double> Jv(T);
	for (int t = 0; t < T; t++) {
		xiycurr[t] = 0.0;
		xivcurr[t] = 0.0;
		xiccurr[t] = ZeroF;
		Ncurr[t] = ZeroI;
		Jy[t] = Ncurr[t][0] * xiycurr[t] + Ncurr[t][2] * xiccurr[t][0];
		Jv[t] = Ncurr[t][1] * xivcurr[t] + Ncurr[t][2] * xiccurr[t][1];
		Vcurr[t] = pow(Y.at(t + 1) - Y.at(t) - mu.back() * Delta, 2);
		Bcurr[t] = 1.0;
	}
	Vcurr[T] = 0.0001;
	xiy.push_back(xiycurr);
	xiv.push_back(xivcurr);
	xic.push_back(xiccurr);
	N.push_back(Ncurr);
	V.push_back(Vcurr);
	B.push_back(Bcurr);
	for (int i = 1; i <= M; i++){
		mu.push_back(musim(kappa.back(), theta.back(), sigmav.back(), rho.back(), Y, Vcurr, Jy, Jv));
		theta.push_back(thetasim(mu.back(), kappa.back(), sigmav.back(), rho.back(), Y, Vcurr, Jy, Jv));
		kappa.push_back(kappasim(mu.back(), theta.back(), sigmav.back(), rho.back(), Y, Vcurr, Jy, Jv));
		//sigvrhosim = sigmavrhosim(mu.back(), kappa.back(), theta.back(), Y, Vcurr, Jy, Jv);
		phi = phivsim(mu.back(), kappa.back(), theta.back(), omega2, Y, Vcurr, Jy, Jv);
		omega2 = omega2sim(mu.back(), kappa.back(), theta.back(), phi, Y, Vcurr, Jy, Jv);
		//sigmav.push_back(sigvrhosim(0));
		//rho.push_back(sigvrhosim(1));
		sigmav.push_back(sqrt(pow(phi, 2) + omega2));
		rho.push_back(phi / sigmav.back());
		etassim = etasim(xiycurr, sqrt(etayplus.back() / etayminus.back()), 2.0, 4.0);
		etayplus.push_back(etassim(0));
		etayminus.push_back(etassim(1));
		etassim = etasim(xivcurr, sqrt(etavplus.back() / etavminus.back()), 3.0, 1.0);
		etavplus.push_back(etassim(0));
		etavminus.push_back(etassim(1));
		muc.push_back(mucsim(nuc.back(), xiccurr, Bcurr));
		nuc.push_back(nucsim(muc.back(), xiccurr, Bcurr));
		lambda.push_back(lambdasim(Ncurr));
		for (int t = 0; t < T; t++) {
			double Yt[2];
			double Vt[2];
			Yt[0] = Y[t];
			Yt[1] = Y[t + 1];
			Vt[0] = Vcurr[t];
			Vt[1] = Vcurr[t + 1];
			xiycurr[t] = xisim(mu.back(), kappa.back(), theta.back(), sigmav.back(), rho.back(), etayplus.back(), etayminus.back(), Yt, Vt, Ncurr[t][0]);
			xivcurr[t] = xisim(mu.back(), kappa.back(), theta.back(), sigmav.back(), rho.back(), etavplus.back(), etavminus.back(), Yt, Vt, Ncurr[t][1]);
			xiccurr[t] = xicsim(mu.back(), kappa.back(), theta.back(), sigmav.back(), rho.back(), muc.back(), nuc.back(), Yt, Vt, Ncurr[t][2], Bcurr[t]);
			Ncurr[t] = Nsim(mu.back(), kappa.back(), theta.back(), sigmav.back(), rho.back(), Yt, Vt, xiycurr[t], xivcurr[t], xiccurr[t], lambda.back());
			Jy[t] = Ncurr[t][0] * xiycurr[t] + Ncurr[t][2] * xiccurr[t][0];
			Jv[t] = Ncurr[t][1] * xivcurr[t] + Ncurr[t][2] * xiccurr[t][1];
			Vcurr[t] = Vsim(mu.back(), kappa.back(), theta.back(), sigmav.back(), rho.back(), Yt, Vt, Jy[t], Jv[t], t);
			Bcurr[t] = Bsim(xiccurr[t], muc.back(), nuc.back(), Bcurr[t]);
		}
		xiy.push_back(xiycurr);
		xiv.push_back(xivcurr);
		xic.push_back(xiccurr);
		N.push_back(Ncurr);
		V.push_back(Vcurr);
		B.push_back(Bcurr);
	}
	ofstream pfile("paramoutput.txt");
		for (int i = 0; i <= sims; i++) {
			pfile << mu[i] << kappa[i] << theta[i] << sigmav[i] << rho[i] << etayplus[i] << etayminus[i] << etavplus[i] << etavminus[i] << muc[i] << nuc[i] << lambda[i];
		}
		pfile.close();

	ofstream xfile("xioutput.txt");
		for (int i = 0; i < sims; i++) {
			for (int t = 0; t < T; t++) {
				xfile << xiy[i][t] << xiv[i][t] << xic[i][t];
			}
		}
		xfile.close();

	ofstream nfile("Noutput.txt");
		for (int i = 0; i < sims; i++) {
			for (int t = 0; t < T; t++) {
				nfile << N[i][t];
			}
		}
		nfile.close();

	ofstream vfile("Voutput.txt");
		for (int i = 0; i < sims; i++) {
			for (int t = 0; t <= T; t++) {
				vfile << V[i][t];
			}
		}
		vfile.close();
	return 0;
}
