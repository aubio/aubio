using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Windows.Media.MediaProperties;
using Windows.Media.Transcoding;
using Windows.Storage;
using Windows.Storage.Pickers;
using Windows.Storage.Streams;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Wav;

namespace AubioComponent.Example
{
    public sealed partial class MainPage
    {
        public MainPage()
        {
            InitializeComponent();
        }

        private async void Button_Click(object sender, RoutedEventArgs e)
        {
            #region setup/show progress

            IProgress<double> progress1 = new Progress<double>(d => ProgressBar.Value = d);
            IProgress<string> progress2 = new Progress<string>(s => ProgressText.Text = s);
            var button = (Button) sender;
            button.Visibility = Visibility.Collapsed;
            ProgressPanel.Visibility = Visibility.Visible;

            #endregion

            #region pick an audio file

            var picker = new FileOpenPicker();
            picker.FileTypeFilter.Add(@".wav");
            picker.FileTypeFilter.Add(@".mp3");
            picker.FileTypeFilter.Add(@".flac");
            picker.SuggestedStartLocation = PickerLocationId.MusicLibrary;
            var file = await picker.PickSingleFileAsync();
            if (file == null) return;

            #endregion

            #region transcode to mono

            var transcoder = new MediaTranscoder();
            var source = await file.OpenAsync(FileAccessMode.ReadWrite);
            var destination = new InMemoryRandomAccessStream();
            var profile = GetMediaEncodingProfile();
            var result = await transcoder.PrepareStreamTranscodeAsync(source, destination, profile);
            if (!result.CanTranscode) return;
            progress2.Report("Transcoding");
            var task = result.TranscodeAsync().AsTask(progress1);
            await task;
            destination.Seek(0);

            #endregion

            #region get tempo

            progress2.Report("Detecting tempo");
            var tempoResults = await Task.Run(() =>
            {
                var results = new List<AubioTempoResult>();
                var windowSize = 1024u;
                var hopSize = 256u;
                var buffer = new float[hopSize];
                using (var stream = destination.CloneStream().AsStream())
                using (var wav = WavFile.FromStream(stream))
                using (var tempo = new AubioTempo("default", windowSize, hopSize, wav.SampleRate))
                {
                    var blocks = (int) Math.Ceiling((double) wav.Length/hopSize);
                    var percent = 0;
                    for (var i = 0; i < blocks; i++)
                    {
                        var read = wav.Read(buffer, (int) hopSize);
                        tempo.SetData(buffer, (uint) read);
                        var data = tempo.GetData();
                        if (data != null) results.Add(data);

                        var percent1 = (int) Math.Ceiling((double) i/blocks*100.0d);
                        if (percent1 <= percent) continue;
                        progress1.Report(percent1);
                        percent = percent1;
                    }
                }
                return results;
            });

            #endregion

            #region get onsets

            progress2.Report("Detecting onsets");
            var onsetResults = await Task.Run(() =>
            {
                var results = new List<AubioOnsetResult>();
                var windowSize = 1024u;
                var hopSize = 256u;
                var buffer = new float[hopSize];
                using (var stream = destination.CloneStream().AsStream())
                using (var wav = WavFile.FromStream(stream))
                using (var onset = new AubioOnset(AubioOnsetMethod.EnergyBased, windowSize, hopSize, wav.SampleRate))
                {
                    var blocks = (int) Math.Ceiling((double) wav.Length/hopSize);
                    var percent = 0;
                    for (var i = 0; i < blocks; i++)
                    {
                        var read = wav.Read(buffer, (int) hopSize);
                        onset.SetData(buffer, (uint) read);
                        var data = onset.GetData();
                        if (data != null) results.Add(data);

                        var percent1 = (int) Math.Ceiling((double) i/blocks*100.0d);
                        if (percent1 <= percent) continue;
                        progress1.Report(percent1);
                        percent = percent1;
                    }
                }
                return results;
            });

            #endregion

            #region done

            source.Dispose();
            destination.Dispose();
            progress1.Report(0.0d);
            progress2.Report("All done, see tabs for results.");
            ListViewTempo.ItemsSource = tempoResults;
            ListViewOnset.ItemsSource = onsetResults;

            #endregion
        }

        private static MediaEncodingProfile GetMediaEncodingProfile()
        {
            var bitsPerSample = 16u;
            var sampleRate = 44100u;
            var channelCount = 1u;
            var bitrate = sampleRate*channelCount*bitsPerSample;
            var subtype = MediaEncodingSubtypes.Pcm;
            var profile = new MediaEncodingProfile
            {
                Audio = new AudioEncodingProperties
                {
                    SampleRate = sampleRate,
                    Bitrate = bitrate,
                    BitsPerSample = bitsPerSample,
                    ChannelCount = channelCount,
                    Subtype = subtype
                },
                Container = new ContainerEncodingProperties
                {
                    Subtype = MediaEncodingSubtypes.Wave
                },
                Video = null
            };
            return profile;
        }
    }
}