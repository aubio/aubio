#pragma once

#include "types.h"
#include "fvec.h"
#include "tempo\tempo.h"
#include "onset\onset.h"
#include "pitch\pitch.h"

namespace AubioComponent
{
	public ref class AubioTempoResult sealed
	{

	private:
		
		uint32 samples_;
		float32 miliseconds_;
		float32 seconds_;
		float32 bpm_;

	internal:
		
		AubioTempoResult(uint32 samples, float32 miliseconds, float32 seconds, float32 bpm) :
			samples_(samples),
			miliseconds_(miliseconds),
			seconds_(seconds),
			bpm_(bpm)
		{

		}

	public:
		
		property uint32 Samples
		{
			uint32 get()
			{
				return samples_;
			}
		}

		property float32 Miliseconds
		{
			float32 get()
			{
				return miliseconds_;
			}
		}

		property float32 Seconds
		{
			float32 get()
			{
				return seconds_;
			}
		}

		property float32 Bpm
		{
			float32 get()
			{
				return bpm_;
			}
		}

	};

	public ref class AubioTempo sealed
	{

	private:
		
		aubio_tempo_t* tempo_;
		fvec_t* in_;
		fvec_t* out_;
		bool disposed_;

	public:

		AubioTempo(Platform::String^ method, uint32 windowSize, uint32 hopSize, uint32 sampleRate) :
			disposed_(false)
		{
			std::wstring wstr(method->Begin());
			std::string str(wstr.begin(), wstr.end());
			char* pMethod = (char*)str.c_str();
			tempo_ = new_aubio_tempo(pMethod, windowSize, hopSize, sampleRate);
			in_ = new_fvec(hopSize);
			out_ = new_fvec(2);
		}

		void SetData(const Platform::Array<float32>^ buffer, uint32 count)
		{
			auto length = min(buffer->Length, count);
			for (auto i = 0u; i < length; i++)
			{
				auto data = buffer[i];
				fvec_set_sample(in_, data, i);
			}
			aubio_tempo_do(tempo_, in_, out_);
		}

		AubioTempoResult^ GetData()
		{
			auto sample = fvec_get_sample(out_, 0);
			if (sample == 0.0f) return nullptr;

			auto samples = aubio_tempo_get_last(tempo_);
			auto miliseconds = aubio_tempo_get_last_ms(tempo_);
			auto seconds = aubio_tempo_get_last_s(tempo_);
			auto bpm = aubio_tempo_get_bpm(tempo_);
			auto result = ref new AubioTempoResult(samples, miliseconds, seconds, bpm);
			return result;
		}

		virtual ~AubioTempo()
		{
			if (disposed_) return;

			del_aubio_tempo(tempo_);
			del_fvec(in_);
			del_fvec(out_);
			tempo_ = nullptr;
			in_ = nullptr;
			out_ = nullptr;

			disposed_ = true;
		}

	};

	public enum class AubioOnsetMethod
	{
		EnergyBased,
		HighFrequencyContent,
		ComplexDomain,
		PhaseBased,
		SpectralDifference,
		KullbackLiebler,
		ModifiedKullbackLiebler,
		SpectralFlux
	};

	public ref class AubioOnsetResult sealed
	{

	private:
		
		uint32 samples_;
		float32 miliseconds_;
		float32 seconds_;

	internal:
		
		AubioOnsetResult(uint32 samples, float32 miliseconds, float32 seconds) :
			samples_(samples),
			miliseconds_(miliseconds),
			seconds_(seconds)
		{

		}

	public:
		
		property uint32 Samples
		{
			uint32 get()
			{
				return samples_;
			}
		}

		property float32 Miliseconds
		{
			float32 get()
			{
				return miliseconds_;
			}
		}

		property float32 Seconds
		{
			float32 get()
			{
				return seconds_;
			}
		}

	};

	public ref class AubioOnset sealed
	{

	private:

		aubio_onset_t* onset_;
		fvec_t* in_;
		fvec_t* out_;
		bool disposed_;

	public:

		AubioOnset(AubioOnsetMethod method, uint32 windowSize, uint32 hopSize, uint32 sampleRate) :
			disposed_(false)
		{
			using namespace Platform;
			using namespace Platform::Collections;
			Map<AubioOnsetMethod, String^>^ map = ref new Map<AubioOnsetMethod, String^>();
			map->Insert(AubioOnsetMethod::EnergyBased, "energy");
			map->Insert(AubioOnsetMethod::HighFrequencyContent, "hfc");
			map->Insert(AubioOnsetMethod::ComplexDomain, "complex");
			map->Insert(AubioOnsetMethod::PhaseBased, "phase");
			map->Insert(AubioOnsetMethod::SpectralDifference, "specdiff");
			map->Insert(AubioOnsetMethod::KullbackLiebler, "kl");
			map->Insert(AubioOnsetMethod::ModifiedKullbackLiebler, "mkl");
			map->Insert(AubioOnsetMethod::SpectralFlux, "specflux");

			auto name = map->Lookup(method);
			std::wstring wstr(name->Begin());
			std::string str(wstr.begin(), wstr.end());
			char* pChar = (char*)str.c_str();

			onset_ = new_aubio_onset(pChar, windowSize, hopSize, sampleRate);
			in_ = new_fvec(hopSize);
			out_ = new_fvec(2);
		}

		void SetData(const Platform::Array<float32>^ buffer, uint32 count)
		{
			auto length = min(buffer->Length, count);
			for (auto i = 0u; i < length; i++)
			{
				auto data = buffer[i];
				fvec_set_sample(in_, data, i);
			}
			aubio_onset_do(onset_, in_, out_);
		}

		AubioOnsetResult^ GetData()
		{
			auto sample = fvec_get_sample(out_, 0);
			if (sample == 0.0f) return nullptr;

			auto samples = aubio_onset_get_last(onset_);
			auto miliseconds = aubio_onset_get_last_ms(onset_);
			auto seconds = aubio_onset_get_last_s(onset_);
			auto result = ref new AubioOnsetResult(samples, miliseconds, seconds);
			return result;
		}

		virtual ~AubioOnset()
		{
			if (disposed_) return;

			del_aubio_onset(onset_);
			del_fvec(in_);
			del_fvec(out_);
			onset_ = nullptr;
			in_ = nullptr;
			out_ = nullptr;

			disposed_ = true;
		}
	};

	public enum class AubioPitchMethod
	{
		SchmittTrigger,
		FastHarmonicComb,
		MultipleComb,
		Yin,
		YinFFT,
		Default = YinFFT,
	};

	public ref class AubioPitchResult sealed
	{

	private:

		float32 pitch_;
		float32 confidence_;

	internal:
		
		AubioPitchResult(float32 pitch, float32 confidence) :
			pitch_(pitch),
			confidence_(confidence)
		{

		}

	public:

		property float32 Pitch
		{
			float32 get()
			{
				return pitch_;
			}
		}

		property float32 Confidence
		{
			float32 get()
			{
				return confidence_;
			}
		}

	};

	public enum class AubioPitchUnit
	{
		Frequency,
		Midi,
		Cent,
		Bin,
		Default
	};

	public ref class AubioPitch sealed
	{

	private:

		aubio_pitch_t * pitch_;
		fvec_t* in_;
		fvec_t* out_;
		AubioPitchUnit unit_;
		float tolerance_;
		bool disposed_;

	public:

		property float SilenceThreshold
		{
			float get()
			{
				return aubio_pitch_get_silence(pitch_);
			}

			void set(float value)
			{
				auto b = aubio_pitch_set_silence(pitch_, value);
				if (b)
				{
					throw ref new Platform::InvalidArgumentException("Could not set silence detection threshold.");
				}
			}
		}

		property float ToleranceThreshold
		{
			float get()
			{
				return tolerance_;
			}

			void set(float value)
			{
				auto b = aubio_pitch_set_tolerance(pitch_, value);
				if (b)
				{
					throw ref new Platform::InvalidArgumentException("Could not set silence tolerance threshold.");
				}

				tolerance_ = value;
			}
		}

		property AubioPitchUnit Unit
		{
			AubioPitchUnit get()
			{
				return unit_;
			}

			void set(AubioPitchUnit value)
			{
				using namespace Platform;
				using namespace Platform::Collections;
				Map<AubioPitchUnit, String^>^ map = ref new Map<AubioPitchUnit, String^>();
				map->Insert(AubioPitchUnit::Bin, "bin");
				map->Insert(AubioPitchUnit::Cent, "cent");
				map->Insert(AubioPitchUnit::Default, "default");
				map->Insert(AubioPitchUnit::Frequency, "freq");
				map->Insert(AubioPitchUnit::Midi, "midi");

				auto name = map->Lookup(value);
				std::wstring wstr(name->Begin());
				std::string str(wstr.begin(), wstr.end());
				char* pChar = (char*)str.c_str();

				auto b = aubio_pitch_set_unit(pitch_, pChar);
				if (b)
				{
					throw ref new Platform::InvalidArgumentException("Could not set pitch detection unit.");
				}

				unit_ = value;
			}
		}

		AubioPitch(AubioPitchMethod method, uint32 windowSize, uint32 hopSize, uint32 sampleRate) :
			disposed_(false)
		{
			using namespace Platform;
			using namespace Platform::Collections;
			Map<AubioPitchMethod, String^>^ map = ref new Map<AubioPitchMethod, String^>();
			map->Insert(AubioPitchMethod::Default, "default");
			map->Insert(AubioPitchMethod::SchmittTrigger, "schmitt");
			map->Insert(AubioPitchMethod::FastHarmonicComb, "fcomb");
			map->Insert(AubioPitchMethod::MultipleComb, "mcomb");
			map->Insert(AubioPitchMethod::Yin, "yin");
			map->Insert(AubioPitchMethod::YinFFT, "yinfft");

			auto name = map->Lookup(method);
			std::wstring wstr(name->Begin());
			std::string str(wstr.begin(), wstr.end());
			char* pChar = (char*)str.c_str();

			pitch_ = new_aubio_pitch(pChar, windowSize, hopSize, sampleRate);
			in_ = new_fvec(hopSize);
			out_ = new_fvec(1);

			Unit = AubioPitchUnit::Default;
		}

		void SetData(const Platform::Array<float32>^ buffer, uint32 count)
		{
			auto length = min(buffer->Length, count);
			for (auto i = 0u; i < length; i++)
			{
				auto data = buffer[i];
				fvec_set_sample(in_, data, i);
			}
			aubio_pitch_do(pitch_, in_, out_);
		}

		AubioPitchResult^ GetData()
		{
			auto pitch = fvec_get_sample(out_, 0);
			auto confidence = aubio_pitch_get_confidence(pitch_);
			auto result = ref new AubioPitchResult(pitch, confidence);
			return result;
		}

		virtual ~AubioPitch()
		{
			if (disposed_) return;

			del_aubio_pitch(pitch_);
			del_fvec(in_);
			del_fvec(out_);
			pitch_ = nullptr;
			in_ = nullptr;
			out_ = nullptr;

			disposed_ = true;
		}
	};
}
