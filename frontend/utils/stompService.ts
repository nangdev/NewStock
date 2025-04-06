import { Client, IMessage, StompSubscription } from '@stomp/stompjs';

type MessageCallback = (message: IMessage) => any;

class StompService {
  private client: Client | null = null;
  private isConnected: boolean = false;
  private subscriptions: Map<string, StompSubscription> = new Map();
  public prefix: string = '/topic/rtp'

  public connect() {
    if (this.client && this.isConnected) return;

    this.client = new Client({
      webSocketFactory: () => new WebSocket('ws://j12a304.p.ssafy.io:8080/api/ws'),
      debug: (msg) => console.log('STOMP:', msg),
      onConnect: (frame) => {
        this.isConnected = true;
      },
      forceBinaryWSFrames: true,
      appendMissingNULLonIncoming: true,
    })

    this.client.activate();
  }


  public subscribe(topic: string, onMessage: MessageCallback) {
    if (!this.client || !this.isConnected) return;
    if (this.subscriptions.has(topic)) return;
    console.log(`subscribe >> ${topic}`)
    const sub = this.client.subscribe(`${this.prefix}/${topic}`, onMessage);
    this.subscriptions.set(topic, sub);
  }

  public unsubscribe(topic: string) {
    if (!this.client || !this.isConnected) return;
    if (!this.subscriptions.has(topic)) return;
    console.log(`unsubscribe >> ${topic}`)
    const unsub = this.client.unsubscribe(`${this.prefix}/${topic}/`);
    this.subscriptions.delete(topic);
  }

  public unsubscribeAll() {
    if (!this.client || !this.isConnected) return;
  
    for (const topic of this.subscriptions.keys()) {
      this.unsubscribe(topic);
    }
    console.log(`unsubscribe all topics`)
    this.subscriptions.clear();
  }
  

  public disconnect() {
    if (!this.client || !this.isConnected) return;
    this.isConnected = false;
    this.subscriptions.clear();
    this.client.deactivate();
  }

  public isReady(): boolean {
    return this.isConnected;
  }
}

export default new StompService();
