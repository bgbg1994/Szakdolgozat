#ifndef _COLOR_3D_H_
#define _COLOR_3D_H_

#include <cassert>

class Color3d {
public:
	double v[3];

	// �f�t�H���g�R���X�g���N�^
	Color3d();

	// �R���X�g���N�^
	Color3d(double r, double g, double b);

	// �R�s�[�R���X�g���N�^
	Color3d(const Color3d& c3d);

	// ���Z�q =
	Color3d& operator=(const Color3d& c3d);

	// �A�N�Z�X���Z�q
	double& operator()(int i);

	// ���Z�q +
	Color3d operator+(const Color3d& c3d);

	// ���Z�q -
	Color3d operator-(const Color3d& c3d);

	// ���Z�q *
	Color3d operator*(const Color3d& c3d);

	// ���Z�q <
	bool operator<(const Color3d& c3d) const;

	// ���Z�q >
	bool operator>(const Color3d& c3d) const;

	/* ���\�b�h��` */
	// �l�̒萔�{
	Color3d multiply(double d);

	// �l�̒萔��
	Color3d divide(double d);
};

#endif
