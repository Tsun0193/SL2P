import {Component, OnInit} from '@angular/core';
import {Store} from '@ngxs/store';
import {VideoStateModel} from '../../../core/modules/ngxs/store/video/video.state';
import {InputMode, SignWritingObj} from '../../../modules/translate/translate.state';
import {
  CopySpokenLanguageText,
  SetSignWritingText,
  SetSpokenLanguageText,
} from '../../../modules/translate/translate.actions';
import {HttpClient} from '@angular/common/http';
import {Observable} from 'rxjs';

const FAKE_WORDS = [
  {
    time: 0.618368,
    sw: ['M507x523S15a28494x496'],
    text: 'B',
  },
  {
    time: 0.876432,
    sw: ['M507x523S15a28494x496S26500493x477'],
    text: 'Your',
  },

  {
    time: 1.102468,
    sw: ['M507x523S15a28494x496S26500493x477', 'M522x525S11541498x491S115494'],
    text: 'Your h',
  },
  {
    time: 1.102468,
    sw: ['M507x523S15a28494x496S26500493x477', 'M522x525S11541498x491'],
    text: 'Your h',
  },
  {
    time: 1.438297,
    sw: ['M507x523S15a28494x496S26500493x477', 'M522x525S11541498x491S11549479x498'],
    text: 'Your',
  },
  {
    time: 1.628503,
    sw: ['M507x523S15a28494x496S26500493x477', 'M522x525S11541498x491S11549479x498S20500489x476'],
    text: 'Your',
  },
  {
    time: 1.786967,
    sw: ['M507x523S15a28494x496S26500493x477', 'M522x525S11541498x491S11549479x498S20600489x476'],
    text: 'Your name',
  },
  {
    time: 1.993408,
    sw: [
      'M507x523S15a28494x496S26500493x477',
      'M522x525S11541498x491S11549479x498S20600489x476',
      'M554x585S30a00481x488S14c39465x545S14c31508x546',
    ],
    text: 'Your name',
  },
  {
    time: 2.163386,
    sw: [
      'M507x523S15a28494x496S26500493x477',
      'M522x525S11541498x491S11549479x498S20600489x476',
      'M554x585S30a00481x488S30300481x477S14c31508x546S14c39465x545S26506539x545S26512445x545',
    ],
    text: 'Your name',
  },
  {
    time: 3.113322,
    sw: [
      'M507x523S15a28494x496S26500493x477',
      'M522x525S11541498x491S11549479x498S20600489x476',
      'M554x585S30a00481x488S30300481x477S14c31508x546S14c39465x545S27102539x545S27116445x545',
    ],
    text: 'What is your name?',
  },
];

@Component({
  selector: 'app-signed-to-spoken',
  templateUrl: './signed-to-spoken.component.html',
  styleUrls: ['./signed-to-spoken.component.scss'],
})
export class SignedToSpokenComponent implements OnInit {
  videoState$!: Observable<VideoStateModel>;
  inputMode$!: Observable<InputMode>;
  spokenLanguage$!: Observable<string>;
  spokenLanguageText$!: Observable<string>;
  hasSent: boolean = false;
  translateResult: string = null;

  constructor(private http: HttpClient, private store: Store) {
    this.videoState$ = this.store.select<VideoStateModel>(state => state.video);
    this.inputMode$ = this.store.select<InputMode>(state => state.translate.inputMode);
    this.spokenLanguage$ = this.store.select<string>(state => state.translate.spokenLanguage);
    this.spokenLanguageText$ = this.store.select<string>(state => state.translate.spokenLanguageText);

    this.store.dispatch(new SetSpokenLanguageText(''));
  }

  ngOnInit(): void {
    // To get the fake translation
    let lastArray = [];
    let lastText = '';

    const f = async () => {
      const video: HTMLVideoElement | null = document.querySelector("div[id='video-container']>video");
      
      if (video !== undefined && video !== null && video.getAttribute("src") !== undefined && 
          video.getAttribute("src") !== null && video.getAttribute("src") !== "") {
        
        if (!this.hasSent) {
          let blob = await fetch(video.getAttribute("src")).then(r => r.blob());
          if (blob !== undefined && blob !== null) {
            this.hasSent = true
            console.log(blob);

            let formData:FormData = new FormData();
            let file = new File([blob], "vid");
            formData.append('uploadFile', file, file.name)

            this.http.post("http://localhost:8000/test", formData).subscribe((data: any) => {
              this.translateResult = data.name;             
            })
          }
          
        }
        
        let resultArray = [];
        let resultText = '';
        for (const step of FAKE_WORDS) {
          if (step.time <= video.currentTime) {
            resultText = step.text;
            resultArray = step.sw;
          }
        }

        if (resultText !== lastText) {
          this.store.dispatch(new SetSpokenLanguageText(resultText));
          lastText = resultText;
        }

        if (JSON.stringify(resultArray) !== JSON.stringify(lastArray)) {
          this.store.dispatch(new SetSignWritingText(resultArray));
          lastArray = resultArray;
        }

      }

      requestAnimationFrame(f);
    };
    f();
  }

  copyTranslation() {
    this.store.dispatch(CopySpokenLanguageText);
  }
}