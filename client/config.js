var config = {
    address: "localhost",
    port: 8080,
    ipWhitelist: ["127.0.0.1", "::ffff:127.0.0.1", "::1"],
    language: "ko",
    timeFormat: 12,
    units: "metric",
  
    modules: [
      {
        module: "alert",
      },
      {
        module: "updatenotification",
        position: "top_bar"
      },
      {
        module: "clock",
        position: "top_left",
        config: {
          dateFormat: "LL dddd",
        }
      },
      {
        module: "calendar",
        position: "top_left",
        config: {
          urgency: 21,
          fetchInterval: 3600000,
          calendars: [{
            url: 'https://calendar.google.com/calendar/ical/ko.south_korea%23holiday%40group.v.calendar.google.com/public/basic.ics',
            symbol: '대한민국 공휴일',
          }]
        }
      },
      {
        module: "currentweather",
        position: "top_right",
        config: {
          location: "Incheon, KR", //https://openweathermap.org/ 에서 지역을 찾으세요.
          locationID: "",
          appid: "53e6cb6592afd9ca6f582741d248f168" //openweathermap.org의 API key를 입력하세요.
        }
      },
      {
        module: "weatherforecast",
        position: "top_right",
        config: {
          location: "Incheon, KR", //https://openweathermap.org/ 에서 지역을 찾으세요.
          locationID: "",
          appid: "53e6cb6592afd9ca6f582741d248f168" //openweathermap.org의 API key를 입력하세요.
        }
      },
      {
        module: "compliments",
        position: "lower_third",
        config: {
          compliments: {
            anytime: [ //아무때나
              "오늘도 좋은 하루!"
            ],
            morning: [ //아침
              "좋은 아침!",
              "힘찬 아침!",
              "잘 잤나요?"
            ],
            afternoon: [ //오후
              "안녕하세요!",
              "멋져요!",
              "잘 지내고 있나요!"
            ],
            evening: [ //저녁
              "와우! 잘 지냈나요?",
              "멋져보이네요!",
              "반가워요!"
            ],
            day_sunny: [//맑은 낮
  
            ],
            day_cloudy: [//흐린 낮
  
            ],
            cloudy: [//흐림
  
            ],
            cloudy_windy: [//흐리고 바람
  
            ],
            showers: [//소나기
  
            ],
            rain: [//비
  
            ],
            thunderstorm: [//천둥번개
  
            ],
            snow: [//눈
  
            ],
            fog: [//안개
  
            ],
            night_clear: [//맑은 밤
  
            ],
            night_cloudy: [//흐린 밤
  
            ],
            night_showers: [//소나기 밤
  
            ],
            night_rain: [//비오는 밤
  
            ],
            night_thunderstorm: [//천둥번개 밤
  
            ],
            night_snow: [//눈오는 밤
  
            ],
            night_alt_cloudy_windy: [//흐리고 바람부는 밤
  
            ],
          }
        }
      },
      {
        module: "MMM-AssistantMk2",
        position: "top_left",
        config: {
          ui: "Fullscreen",
          assistantConfig: {
            latitude: 37.57,
            longitude: 126.98
          },
          micConfig: {
            recoder : "arecord",  
            device  : "plughw:1",
          },
          recipes: [ 
            "with-MMM-Hotword.js",
            "with-MMM-Youtube.js",
            ],
          profiles: {
            "default": {
              lang: "ko-KR"
            }
          },
          addons: false,
        }
      },
      {
        module: "MMM-YouTube",
        position: "top_right"
      },
      {
        module: "MMM-Hotword",
        position: "top_center",
        config: {
          chimeOnFinish: null,
          recipes: [ "with-AMk2v3_smart-mirror.js" ],
          mic: {
            recordProgram: "arecord",
            device: "plughw:1"
          }
        }
      },
    ]
  };
  
  /*************** DO NOT EDIT THE LINE BELOW ***************/
  if (typeof module !== "undefined") {
    module.exports = config;
  }