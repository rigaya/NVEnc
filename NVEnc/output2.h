//----------------------------------------------------------------------------------
//	�o�̓v���O�C�� �w�b�_�[�t�@�C�� for AviUtl ExEdit2
//	By �j�d�m����
//----------------------------------------------------------------------------------

// �o�͏��\����
struct OUTPUT_INFO {
	int flag;			//	�t���O
	static constexpr int FLAG_VIDEO = 1; // �摜�f�[�^����
	static constexpr int FLAG_AUDIO = 2; // �摜�f�[�^����
	int w, h;			//	�c���T�C�Y
	int rate, scale;	//	�t���[�����[�g�A�X�P�[��
	int n;				//	�t���[����
	int audio_rate;		//	�����T���v�����O���[�g
	int audio_ch;		//	�����`�����l����
	int audio_n;		//	�����T���v�����O��
	LPCWSTR savefile;	//	�Z�[�u�t�@�C�����ւ̃|�C���^

	// DIB�`���̉摜�f�[�^���擾���܂�
	// frame	: �t���[���ԍ�
	// format	: �摜�t�H�[�}�b�g
	//			  0(BI_RGB) = RGB24bit / 'P''A''6''4' = PA64 / 'H''F''6''4' = HF64 / 'Y''U''Y''2' = YUY2 / 'Y''C''4''8' = YC48
	// ��PA64��DXGI_FORMAT_R16G16B16A16_UNORM(��Z�ς݃�)�ł�
	// ��HF64��DXGI_FORMAT_R16G16B16A16_FLOAT(��Z�ς݃�)�ł�
	// ��YC48�͌݊��Ή��̋������t�H�[�}�b�g�ł� 
	// �߂�l	: �f�[�^�ւ̃|�C���^
	//			  �摜�f�[�^�|�C���^�̓��e�͎��ɊO���֐����g�������C���ɏ�����߂��܂ŗL��
	void* (*func_get_video)(int frame, DWORD format);

	// PCM�`���̉����f�[�^�ւ̃|�C���^���擾���܂�
	// start	: �J�n�T���v���ԍ�
	// length	: �ǂݍ��ރT���v����
	// readed	: �ǂݍ��܂ꂽ�T���v����
	// format	: �����t�H�[�}�b�g
	//			  1(WAVE_FORMAT_PCM) = PCM16bit / 3(WAVE_FORMAT_IEEE_FLOAT) = PCM(float)32bit
	// �߂�l	: �f�[�^�ւ̃|�C���^
	//			  �����f�[�^�|�C���^�̓��e�͎��ɊO���֐����g�������C���ɏ�����߂��܂ŗL��
	void* (*func_get_audio)(int start, int length, int* readed, DWORD format);

	// ���f���邩���ׂ܂�
	// �߂�l	: TRUE�Ȃ璆�f
	bool (*func_is_abort)();

	// �c�莞�Ԃ�\�������܂�
	// now		: �������Ă���t���[���ԍ�
	// total	: �������鑍�t���[����
	// �߂�l	: TRUE�Ȃ琬��
	void (*func_rest_time_disp)(int now, int total);

	// �f�[�^�擾�̃o�b�t�@��(�t���[����)��ݒ肵�܂� ���W����4�ɂȂ�܂�
	// �o�b�t�@���̔����̃f�[�^���ǂ݃��N�G�X�g����悤�ɂȂ�܂�
	// video	: �摜�f�[�^�̃o�b�t�@��
	// audio	: �����f�[�^�̃o�b�t�@��
	void (*func_set_buffer_size)(int video_size, int audio_size);
};

// �o�̓v���O�C���\����
struct OUTPUT_PLUGIN_TABLE {
	int flag;				// �t���O �����g�p
	static constexpr int FLAG_VIDEO = 1; //	�摜���T�|�[�g����
	static constexpr int FLAG_AUDIO = 2; //	�������T�|�[�g����
	LPCWSTR name;			// �v���O�C���̖��O
	LPCWSTR filefilter;		// �t�@�C���̃t�B���^
	LPCWSTR information;	// �v���O�C���̏��

	// �o�͎��ɌĂ΂��֐��ւ̃|�C���^
	bool (*func_output)(OUTPUT_INFO* oip);

	// �o�͐ݒ�̃_�C�A���O��v�����ꂽ���ɌĂ΂��֐��ւ̃|�C���^ (nullptr�Ȃ�Ă΂�܂���)
	bool (*func_config)(HWND hwnd, HINSTANCE dll_hinst);

	// �o�͐ݒ�̃e�L�X�g�����擾���鎞�ɌĂ΂��֐��ւ̃|�C���^ (nullptr�Ȃ�Ă΂�܂���)
	// �߂�l	: �o�͐ݒ�̃e�L�X�g���(���Ɋ֐����Ă΂��܂œ��e��L���ɂ��Ă���)
	LPCWSTR (*func_get_config_text)();
};
