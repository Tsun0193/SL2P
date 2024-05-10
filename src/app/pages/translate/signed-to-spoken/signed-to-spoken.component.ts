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
            let file = new File([blob], "vid", { type: blob.type });
            formData.append('uploadFile', file, file.name,)

            this.http.post("http://127.0.0.1:8000/test", formData).subscribe((data: any) => {
              this.translateResult = data.name + " " + data.type;             
            })
          }
          
        }
        
        let resultArray = [];
        let resultText = this.translateResult;
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